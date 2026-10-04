# Dynamic component tables

**Status**: Implemented (experimental prototype)

## Motivation

Heterogeneous simulations need storage proportional to the entities they use.
Madrona keeps component columns contiguous in virtual address space while mapping
physical pages as tables grow. This preserves the pointers exported as learning
framework tensors. Its device-side growth requests use a dedicated host allocator
thread (`src/mw/device/memory.cpp:24-97` and `src/mw/cuda_exec.cpp:1602-1705`);
`src/mw/device/state.cpp:29-80` grows each column and publishes the minimum mapped
row capacity. Entity slots hold generations and locations, while world sorting
compacts deleted rows and remaps surviving slots.

## Design

`warp.experimental.table.DynamicTable` reserves an independent VirtualMemory
region for every component column and persistent compaction scratch array. The
exported array shapes and addresses remain fixed. `ensure_capacity(rows)` commits
all regions before publishing a new logical row capacity. Kernel accesses must
stay within that prefix. Only the prefix is safe to export or copy: calling
`numpy()` on an entire reserved array can touch unmapped pages.

Device `append_row(state, world)` uses an atomic compare-and-swap to reserve a row
without increasing the extent beyond committed capacity. An invalid world or
exhausted capacity returns -1 and records a sticky device error. The caller writes
its components for a successful row; consumption happens in a subsequent kernel
launch. `check_errors()` synchronizes and raises; growth does not silently retry
lost work. Callers retain and retry failed inputs explicitly.

For known per-world creation counts, `append_worlds(counts)` scans the counts in
int64, reserves their combined row range once, and initializes metadata in
parallel. It returns device offsets into the appended rows. The batch either
succeeds completely or returns -1 offsets, preserving existing entities.
Negative counts and sums larger than committed capacity are rejected. This
avoids per-entity contention on the table's reservation counter. Device-side
`append_rows(state, world, count)` supports a smaller producer batch with one
atomic reservation. `clear()` invalidates all surviving handles and reconstructs
the free-slot pool without per-row atomic updates.

Entity handles contain a slot ID and a uint64 generation. `find_row` resolves the
slot to its current row. `delete_entity` atomically marks an entity dead once,
invalidates its slot, increments its generation, and decrements live count.
`compact()` histograms live worlds, scans world counts, scatters destination
indices, gathers components into scratch, and copies them back to the original
columns. It remaps entity slots and rebuilds the reusable slot pool. This preserves
both tensor pointers and surviving handles. Sorting is by world; relative order
within a world is unspecified. World offsets become valid after compaction and
stay valid until the next append or delete phase.

Append, delete, and compact phases must be ordered on one stream; concurrent
mutation across streams is unsupported. Compaction copies every live component
and scans committed entity slots. Component initialization is the caller's
responsibility. The table does not provide autodiff, arbitrary concurrent queries,
archetype migration, or physical shrinking. A deleted handle remains invalid
until uint64 generation wraparound after 2^64 reuse cycles.

Captured launches use a fixed dispatch width and grid-stride loops over device
counts. Logical entity counts and component memberships can therefore change on
replay without rebuilding the graph. Physical pages are committed on the host
between completed replays. This is an explicit host boundary, not GPU demand
faulting or Madrona's synchronous device-to-host growth service. CUDA graph
capture alone does not supply a host service that makes a newly requested row
safe to touch in the current kernel.

## Memory and throughput limits

Each independent reservation commits at least one hardware allocation granule.
The RTX PRO 6000 test device reports a minimum granule of 2 MiB. A table with an
int column and a vec3 column has thirteen regions and initially consumes 26 MiB
plus fixed world metadata and batch scan scratch,
even for one logical row. Grouping small metadata columns in a shared allocation
or sharing page-sized slabs across archetypes would reduce this overhead. This
prototype keeps ownership and growth explicit before introducing pooling.

Persistent scratch approximately doubles component storage. Compaction is linear
in the current row extent, committed entity slots, live component bytes, and
world count. A production ECS should compact only dirty archetypes and use a
GPU radix implementation if deterministic intra-world sorting is required. This
prototype uses existing Warp scan support and bounded atomic world scatter.

The runnable `warp.examples.core.example_dynamic_table` changes 8192 worlds from
single-robot sized columns to mixed robot sizes using the same graph, checks world
ranges, and prints measured ECS maintenance time and physical table memory. It
is an allocation/sorting demonstration, not a heterogeneous MuJoCo solver.
On the RTX PRO 6000 test device, 50 replays averaged 0.078 ms for 57344 rows
and 0.132 ms for 114695 rows, using 22.3 MiB of table-owned mapped memory and
fixed scratch. These measurements include clear, batch append, component
initialization, and world grouping, and exclude physical growth and MuJoCo
physics. Utility scan workspace and the CUDA context are excluded from the
table's memory accounting.

## Validation

For a pure-Warp GPU-only population example, run
`uv run -m warp.examples.core.example_dynamic_particle_worlds --worlds 8192`.
It commits capacity at startup and captures one 32-replay particle lifecycle:
four particles per world, twelve in even worlds while odd worlds continue,
one per world, then mixed populations of three to twelve. Device kernels select
resets, delete entities, compact, batch-append, initialize, and integrate gravity.
Device checks verify surviving handles and particle ages across compaction and
invalidated deleted handles. Readback happens after the rollout. At 8192 worlds,
the four recorded populations contain 32768, 65536, 8192, and 61440 particles;
column addresses and committed storage remain unchanged. This demonstrates
entity storage and simple particle integration, with no MuJoCo dependency.

`uv run warp/tests/test_dynamic_table.py` checks bounded concurrent overflow,
invalid worlds, duplicate deletion, stable handles through world grouping,
recycled-slot generation invalidation, scalar/vector component retention, and
captured graph replay after physical growth past the CUDA allocation granule.
It compares original component pointers before and after growth and compaction.
Bulk tests reject negative counts and int32-overflowing count sums without
partial mutation, and verify handle invalidation after clear. The runnable ECS
example validates changed per-world counts using the same captured graph.

## Resuming the prototype

The Warp checkpoint is on `stafah/dynamic-memory-prototype` in
`StafaH/warp`. Its companion mjwarp checkpoint is on
`stafah/heterogeneous-worlds-prototype` in `StafaH/mujoco_warp`.
Use sibling checkout directories named `mustafa_warp` and `mustafa_mjwarp`;
the mjwarp project's editable dependency resolves `../mustafa_warp`.
Native build outputs are not committed. Rebuild Warp before running the examples:

```bash
uv run build_lib.py --quick
uv run --no-sync -m unittest warp.tests.test_virtual_memory warp.tests.test_capture_scratch warp.tests.test_dynamic_table
uv run --no-sync python -m warp.examples.core.example_dynamic_collision_worlds --worlds 8192 --output collision_worlds.html
```

The quick build requires a driver compatible with the installed CUDA toolkit.
Use `uv run build_lib.py` when a quick build is unsuitable. Physical growth is
still a host boundary between completed graph replays. The native mjwarp contact
integration uses a transient shared column arena and native append counters,
without generic table handles, payload copying, or compaction. Future work should
preserve those distinctions when comparing allocation, packing, and solver costs.
