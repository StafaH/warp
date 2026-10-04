# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-compatible component columns with explicit physical growth boundaries.

Append, deletion, and compaction are separate ordered kernel phases. CUDA physical
memory grows on the host between completed graph replays; kernels never touch an
uncommitted row. Exported column addresses and reserved shapes remain constant.
"""

import operator as _operator
from typing import Any as _Any

import warp as _wp

__all__ = [
    "DynamicTable",
    "Entity",
    "TableState",
    "append_row",
    "append_rows",
    "delete_entity",
    "entity_at",
    "find_row",
]

_wp.set_module_options({"enable_backward": False})


@_wp.struct
class Entity:
    slot: int
    generation: _wp.uint64


@_wp.struct
class TableState:
    # extent, committed row capacity, error bits, live rows, free-list scratch
    counters: _wp.array[int]
    world: _wp.array[int]
    alive: _wp.array[int]
    entity: _wp.array[int]
    slot_to_row: _wp.array[int]
    generation: _wp.array[_wp.uint64]
    free_slots: _wp.array[int]
    num_worlds: int


@_wp.func
def append_rows(state: TableState, world: int, count: int) -> int:
    """Reserve a whole row batch; return -1 on failure without a partial append.

    Write user components after this function returns a nonnegative row. Consumers
    must run in a later kernel launch. Failed appends do not increase row counts.
    """
    if world < 0 or world >= state.num_worlds:
        _wp.atomic_or(state.counters, 2, 2)
        return -1
    if count < 0:
        _wp.atomic_or(state.counters, 2, 4)
        return -1
    row = _wp.atomic_add(state.counters, 0, 0)
    while count <= state.counters[1] - row:
        observed = _wp.atomic_cas(state.counters, 0, row, row + count)
        if observed == row:
            for index in range(row, row + count):
                slot = state.free_slots[index]
                state.entity[index] = slot
                state.world[index] = world
                state.alive[index] = 1
                state.slot_to_row[slot] = index
            _wp.atomic_add(state.counters, 3, count)
            return row
        row = observed
    _wp.atomic_or(state.counters, 2, 1)
    return -1


@_wp.func
def append_row(state: TableState, world: int) -> int:
    """Reserve one row, or return -1 and record a sticky error."""
    return append_rows(state, world, 1)


@_wp.func
def entity_at(state: TableState, row: int) -> Entity:
    entity = Entity()
    entity.slot = state.entity[row]
    entity.generation = state.generation[entity.slot]
    return entity


@_wp.func
def find_row(state: TableState, entity: Entity) -> int:
    """Resolve a stable handle, or return -1 for a deleted/recycled entity."""
    if entity.slot < 0 or entity.slot >= state.counters[1]:
        return -1
    if state.generation[entity.slot] != entity.generation:
        return -1
    return state.slot_to_row[entity.slot]


@_wp.func
def delete_entity(state: TableState, entity: Entity) -> bool:
    """Mark an entity dead once; reclaim its row at the next compaction phase."""
    row = find_row(state, entity)
    if row < 0:
        return False
    if _wp.atomic_cas(state.alive, row, 1, 0) != 1:
        return False
    state.slot_to_row[entity.slot] = -1
    state.generation[entity.slot] += _wp.uint64(1)
    _wp.atomic_sub(state.counters, 3, 1)
    return True


@_wp.kernel
def _initialize_slots(state: TableState, begin: int, end: int, width: int):
    for slot in range(begin + _wp.tid(), end, width):
        state.slot_to_row[slot] = -1
        state.generation[slot] = _wp.uint64(0)
        state.free_slots[slot] = slot
        state.alive[slot] = 0


@_wp.kernel
def _set_capacity(counters: _wp.array[int], capacity: int):
    counters[1] = capacity


@_wp.kernel
def _histogram(state: TableState, counts: _wp.array[int], width: int):
    for row in range(_wp.tid(), state.counters[0], width):
        if state.alive[row] != 0:
            _wp.atomic_add(counts, state.world[row], 1)


@_wp.kernel
def _destinations(
    state: TableState, offsets: _wp.array[int], positions: _wp.array[int], dst: _wp.array[int], width: int
):
    for row in range(_wp.tid(), state.counters[0], width):
        dest = int(-1)
        if state.alive[row] != 0:
            world = state.world[row]
            dest = offsets[world] + _wp.atomic_add(positions, world, 1)
        dst[row] = dest


@_wp.kernel
def _gather(state: TableState, src: _wp.array[_Any], scratch: _wp.array[_Any], dst: _wp.array[int], width: int):
    for row in range(_wp.tid(), state.counters[0], width):
        dest = dst[row]
        if dest >= 0:
            scratch[dest] = src[row]


@_wp.kernel
def _copy_back(src: _wp.array[_Any], scratch: _wp.array[_Any], offsets: _wp.array[int], num_worlds: int, width: int):
    for row in range(_wp.tid(), offsets[num_worlds], width):
        src[row] = scratch[row]


@_wp.kernel
def _remap(state: TableState, offsets: _wp.array[int], width: int):
    count = offsets[state.num_worlds]
    for row in range(_wp.tid(), count, width):
        state.alive[row] = 1
        state.slot_to_row[state.entity[row]] = row


@_wp.kernel
def _finish_compaction(state: TableState, offsets: _wp.array[int]):
    state.counters[0] = offsets[state.num_worlds]
    state.counters[4] = 0


@_wp.kernel
def _recycle_slots(state: TableState, width: int):
    for slot in range(_wp.tid(), state.counters[1], width):
        if state.slot_to_row[slot] == -1:
            index = state.counters[0] + _wp.atomic_add(state.counters, 4, 1)
            state.free_slots[index] = slot


@_wp.kernel
def _clear_errors(counters: _wp.array[int]):
    counters[2] = 0


@_wp.kernel
def _clear_table(state: TableState, width: int):
    for slot in range(_wp.tid(), state.counters[1], width):
        if state.slot_to_row[slot] >= 0:
            state.generation[slot] += _wp.uint64(1)
        state.slot_to_row[slot] = -1
        state.free_slots[slot] = slot
        state.alive[slot] = 0


@_wp.kernel
def _finish_clear(state: TableState):
    state.counters[0] = 0
    state.counters[3] = 0


@_wp.kernel
def _batch_counts(state: TableState, counts: _wp.array[int], wide_counts: _wp.array[_wp.int64], error: _wp.array[int]):
    world = _wp.tid()
    count = counts[world]
    if count < 0:
        _wp.atomic_or(state.counters, 2, 4)
        _wp.atomic_or(error, 0, 1)
        count = 0
    wide_counts[world] = _wp.int64(count)


@_wp.kernel
def _reserve_batch(state: TableState, prefix: _wp.array[_wp.int64], error: _wp.array[int], begin: _wp.array[int]):
    begin[0] = -1
    count = prefix[state.num_worlds]
    if error[0] == 0:
        if count <= _wp.int64(state.counters[1] - state.counters[0]):
            begin[0] = state.counters[0]
            state.counters[0] += int(count)
            state.counters[3] += int(count)
        else:
            _wp.atomic_or(state.counters, 2, 1)


@_wp.kernel
def _initialize_batch(state: TableState, prefix: _wp.array[_wp.int64], begin: _wp.array[int], offsets: _wp.array[int]):
    world = _wp.tid()
    if begin[0] >= 0:
        first = begin[0] + int(prefix[world])
        end = begin[0] + int(prefix[world + 1])
        offsets[world] = first
        for row in range(first, end):
            slot = state.free_slots[row]
            state.entity[row] = slot
            state.world[row] = world
            state.alive[row] = 1
            state.slot_to_row[slot] = row
        if world == 0:
            offsets[state.num_worlds] = begin[0] + int(prefix[state.num_worlds])
    else:
        offsets[world] = -1
        if world == 0:
            offsets[state.num_worlds] = -1


class DynamicTable:
    """Store heterogeneous live entities in contiguous component columns.

    CUDA columns and compaction scratch use ``warp.VirtualMemory``. CPU storage
    eagerly allocates the reservation and is intended for correctness testing.
    ``capacity`` is the logical accessible prefix, rounded physical page sizes
    can be larger. Tables require exclusive use of the selected device stream;
    append, delete, compact, and host growth must not overlap across streams.
    World offsets are refreshed only by ``compact()``. Intra-world order is
    unspecified. Views/tensors export the reserved shape: expose only live rows
    after compaction, and retain the table for their lifetime.

    Args:
        columns: Mapping from component names to scalar, vector, or struct types.
        max_rows: Maximum reserved row count (must fit signed 32-bit indexing).
        num_worlds: Number of valid world IDs.
        initial_capacity: Initially committed accessible row count.
        device: Storage and execution device.
        dispatch_width: Fixed launch width; kernels loop over device row counts.
    """

    def __init__(self, columns, max_rows, num_worlds, initial_capacity=0, device=None, dispatch_width=65536):
        max_rows = _operator.index(max_rows)
        num_worlds = _operator.index(num_worlds)
        initial_capacity = _operator.index(initial_capacity)
        dispatch_width = _operator.index(dispatch_width)
        if not 0 < max_rows < 2**31:
            raise ValueError("max_rows must be between 1 and 2**31 - 1")
        if not 0 < num_worlds < 2**31 - 1:
            raise ValueError("num_worlds must be positive and support an offset sentinel")
        if not 0 <= initial_capacity <= max_rows:
            raise ValueError("initial_capacity must be between zero and max_rows")
        if dispatch_width <= 0:
            raise ValueError("dispatch_width must be positive")
        self.device = _wp.get_device(device)
        self.max_rows = max_rows
        self.capacity = 0
        self.dispatch_width = min(dispatch_width, max_rows)
        self._regions = []
        self._arrays = []
        self._fixed_arrays = []
        self.columns = {name: self._allocate(dtype) for name, dtype in columns.items()}
        self._scratch = {name: self._allocate(dtype) for name, dtype in columns.items()}
        self.state = TableState()
        self.state.counters = self._fixed(5, int)
        self.state.num_worlds = num_worlds
        for name in ("world", "alive", "entity", "slot_to_row", "free_slots"):
            setattr(self.state, name, self._allocate(int))
        self.state.generation = self._allocate(_wp.uint64)
        self._world_scratch = self._allocate(int)
        self._entity_scratch = self._allocate(int)
        self._dest = self._allocate(int)
        self.world_offsets = self._fixed(num_worlds + 1, int)
        self._counts = self._fixed(num_worlds + 1, int)
        self._positions = self._fixed(num_worlds + 1, int)
        self._batch_counts = self._fixed(num_worlds + 1, _wp.int64)
        self._batch_prefix = self._fixed(num_worlds + 1, _wp.int64)
        self._batch_offsets = self._fixed(num_worlds + 1, int)
        self._batch_begin = self._fixed(1, int)
        self._batch_error = self._fixed(1, int)
        self.ensure_capacity(initial_capacity)

    def _fixed(self, size, dtype):
        array = _wp.zeros(size, dtype=dtype, device=self.device)
        self._fixed_arrays.append(array)
        return array

    def _allocate(self, dtype):
        if self.device.is_cuda:
            region = _wp.VirtualMemory(self.max_rows * _wp.types.type_size_in_bytes(dtype), device=self.device)
            array = region.array((self.max_rows,), dtype=dtype)
            self._regions.append((region, _wp.types.type_size_in_bytes(dtype)))
        else:
            array = _wp.empty(self.max_rows, dtype=dtype, device=self.device)
        self._arrays.append(array)
        return array

    @property
    def committed_bytes(self):
        """Physical mapped bytes plus fixed metadata/scratch allocation bytes."""
        fixed = sum(array.capacity for array in self._fixed_arrays)
        if self.device.is_cuda:
            return fixed + sum(region.committed_size for region, _ in self._regions)
        return fixed + sum(array.capacity for array in self._arrays)

    def ensure_capacity(self, rows):
        """Commit rows without moving columns, outside capture/completed replay.

        Growth never retries failed appends: callers must check errors and retry
        their failed work explicitly. Growth does not clear recorded errors.
        """
        rows = _operator.index(rows)
        if not 0 <= rows <= self.max_rows:
            raise ValueError("Requested rows exceed the table reservation")
        if self.device.is_capturing:
            raise RuntimeError("DynamicTable growth is forbidden during graph capture")
        if rows <= self.capacity:
            return
        _wp.synchronize_device(self.device)
        for region, itemsize in self._regions:
            region.commit(rows * itemsize)
        _wp.launch(
            _initialize_slots,
            dim=self.dispatch_width,
            inputs=[self.state, self.capacity, rows, self.dispatch_width],
            device=self.device,
        )
        _wp.launch(_set_capacity, dim=1, inputs=[self.state.counters, rows], device=self.device)
        self.capacity = rows

    def append_worlds(self, counts):
        """Append per-world batches with one reservation; return device row offsets.

        ``counts`` is an int32 array of length ``num_worlds`` on this table's
        device. The whole batch succeeds or fails; returned offsets are -1 on
        failure. On success world ``w`` owns ``[offsets[w], offsets[w+1])``;
        initialize user components in a subsequent kernel. Returned offsets
        are overwritten by the next call. Recordable inside a graph.
        """
        worlds = self.state.num_worlds
        if counts.device != self.device or counts.shape != (worlds,) or counts.dtype != _wp.int32:
            raise ValueError("counts must be an int32 array of num_worlds elements on the table device")
        self._batch_error.zero_()
        _wp.launch(
            _batch_counts,
            dim=worlds,
            inputs=[self.state, counts, self._batch_counts, self._batch_error],
            device=self.device,
        )
        _wp.utils.array_scan(self._batch_counts, self._batch_prefix, inclusive=False)
        _wp.launch(
            _reserve_batch,
            dim=1,
            inputs=[self.state, self._batch_prefix, self._batch_error, self._batch_begin],
            device=self.device,
        )
        _wp.launch(
            _initialize_batch,
            dim=worlds,
            inputs=[self.state, self._batch_prefix, self._batch_begin, self._batch_offsets],
            device=self.device,
        )
        return self._batch_offsets

    def clear(self):
        """Delete all entities and recycle their slots without per-row atomics.

        Invalidates surviving handles and resets world offsets; retains physical
        memory and sticky error flags. Recordable inside a graph.
        """
        _wp.launch(_clear_table, dim=self.dispatch_width, inputs=[self.state, self.dispatch_width], device=self.device)
        _wp.launch(_finish_clear, dim=1, inputs=[self.state], device=self.device)
        self.world_offsets.zero_()

    def compact(self):
        """Compact live rows and group worlds; recordable inside a CUDA graph.

        This gathers into persistent scratch then copies back to original storage,
        preserving exported addresses and entity handles. Cost is linear in live
        extent, committed slots, component bytes, and number of worlds.
        """
        width = self.dispatch_width
        self._counts.zero_()
        self._positions.zero_()
        _wp.launch(_histogram, dim=width, inputs=[self.state, self._counts, width], device=self.device)
        _wp.utils.array_scan(self._counts, self.world_offsets, inclusive=False)
        _wp.launch(
            _destinations,
            dim=width,
            inputs=[self.state, self.world_offsets, self._positions, self._dest, width],
            device=self.device,
        )
        pairs = [(self.state.world, self._world_scratch), (self.state.entity, self._entity_scratch)]
        pairs.extend((array, self._scratch[name]) for name, array in self.columns.items())
        for array, scratch in pairs:
            _wp.launch(_gather, dim=width, inputs=[self.state, array, scratch, self._dest, width], device=self.device)
        for array, scratch in pairs:
            _wp.launch(
                _copy_back,
                dim=width,
                inputs=[array, scratch, self.world_offsets, self.state.num_worlds, width],
                device=self.device,
            )
        _wp.launch(_remap, dim=width, inputs=[self.state, self.world_offsets, width], device=self.device)
        _wp.launch(_finish_compaction, dim=1, inputs=[self.state, self.world_offsets], device=self.device)
        _wp.launch(_recycle_slots, dim=width, inputs=[self.state, width], device=self.device)

    def clear_errors(self):
        """Reset sticky failure flags on the device; recordable inside a graph."""
        _wp.launch(_clear_errors, dim=1, inputs=[self.state.counters], device=self.device)

    def check_errors(self):
        """Synchronize and raise on failed append; existing rows stay valid."""
        errors = int(self.state.counters.numpy()[2])
        if errors & 2:
            raise ValueError("DynamicTable append received an invalid world ID")
        if errors & 4:
            raise ValueError("DynamicTable append received a negative row count")
        if errors & 1:
            raise RuntimeError("DynamicTable committed row capacity exhausted; grow and retry failed appends")

    @property
    def live_count(self):
        """Read the live row count (synchronizes the device)."""
        return int(self.state.counters.numpy()[3])


for _function in (append_rows, append_row, entity_at, find_row, delete_entity):
    _wp._src.context.register_api_function(
        _function, module="warp.experimental.table", python_callable=False, differentiable=False
    )
