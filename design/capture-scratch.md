# Preallocated Capture Scratch

**Status**: Implemented prototype

## Motivation

CUDA conditional body graphs cannot contain memory allocation or free nodes. Existing mjwarp physics functions construct temporary Warp arrays and native CUB scan workspaces while capturing, preventing an entire step from being placed inside a device-selected conditional branch. A fixed, committed scratch buffer allows those existing kernels to be captured without changing their physics implementations.

## Design

`warp.ScopedCaptureScratch(buffer)` accepts an existing contiguous CUDA uint8 array. The scope installs a Python custom allocator for Warp arrays and a thread-local native bump arena for native allocations on capturing streams. Suballocations use 256-byte alignment and bounded capacity. Exhaustion is reported and latched so scope exit rejects a graph even when a native wrapper does not propagate its allocation failure.

Native free functions recognize pointers into live scratch arenas and do not free suballocations. This recognition persists after scope exit. Each scratch-created Warp array retains the allocator through its existing deallocator/allocator references. Captured Warp graphs retain the scope to protect native-only temporary pointers. The scope retains the backing array and its actual memory kind. Metadata is unregistered before the backing owner is released. CPU-only native builds expose inert binding stubs.

`rewind()` resets the bump offset during capture construction only. This permits mutually exclusive model branches or ordered chunks to reuse identical pointer addresses. It does not record a replay-time allocator reset. Applications must ensure those users cannot run concurrently and must not keep old temporary contents as persistent state. Entering the scope does not implicitly rewind, avoiding accidental overlap across unrelated captures.

Scratch allocation and rewind require the device's current stream to be capturing. Alternate-stream captures use the normal `ScopedStream` pattern. APIC serialization, concurrent host allocation on the scoped device, and concurrent graph execution over one backing buffer are unsupported. The supplied buffer must already be fully committed; the primitive performs no runtime physical allocation or VMM mapping.

Native sort already uses retained side-stream workspaces for conditional capture. Those allocations remain managed by its existing graph lifecycle. Native scan uses the scoped arena. Other native allocators routed through Warp's allocation functions use the arena when their current allocation stream is capturing.

## Testing Strategy

Unit tests cover native CUB scan inside a conditional body, mutually exclusive branches that alias storage after rewind, restoration of the previous device allocator, graph-owned native scratch lifetime, escaped array ownership, and CPU rejection. A separate mjwarp proof captures a complete contact-enabled step inside `capture_if`, including SAP segmented sort/scan and the Newton solver's nested conditional loop. Its true/false replay schedule exactly matches eager qpos/qvel updates for 32 worlds, and graph destruction releases the Python scratch owner.
