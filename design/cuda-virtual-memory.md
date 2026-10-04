# CUDA Virtual Memory

**Status**: Implemented prototype

## Motivation

GPU tables need stable contiguous addresses for tensors and captured kernels while growing their physical footprint. Ordinary CUDA allocations require either reserving the full maximum in physical memory or reallocating and updating captured arguments. CUDA driver virtual memory separates address reservation from physical backing.

## Design

`warp.VirtualMemory(size_in_bytes, device, initial_size)` reserves a page-rounded address range with `cuMemAddressReserve`. `commit(bytes)` grows a physically backed prefix using `cuMemCreate`, `cuMemMap`, and `cuMemSetAccess`. The mapped range retains its reference to physical memory after the allocation handle is released. Existing data and the base pointer stay fixed; new pages are cleared before commitment returns.

A reservation exposes contiguous typed `array(shape, dtype, offset)` views. Views retain their owner through Warp's existing `_ref` lifetime mechanism, and explicit close refuses live views. Views may cover the uncommitted tail for graph arguments. Every kernel access must be bounded by an active device-side count; array copies and memset-like operations must use a committed-size view.

Mapping changes synchronize the CUDA context and reject Warp captures or capture on the current stream. CUDA context synchronization also fails safely if an unrecognized external capture prevents a quiescent boundary. GPU kernels cannot call VMM APIs. A graph is captured with maximum launch dimensions and a stable pointer, then replayed with a mutable device count. Growth happens between replay epochs. Applications retain the owner for the lifetime of graphs and borrowed external tensors, as for ordinary Warp arrays.

Destructors defer release during Warp graph capture through the existing native deferred-action lifecycle. Freeing synchronizes prior device work, unmaps every mapping, frees the address reservation, and destroys metadata. CPU-only native builds export inert symbols so normal Warp initialization remains valid; creating a reservation on CPU raises an error.

Minimum granularity is device dependent. The available RTX PRO 6000 Blackwell reports 2 MiB, so separate small table columns cost one 2 MiB page each. Future suballocation can amortize that cost without changing this primitive. This prototype does not implement shrinking, fragmentation reuse, peer access, IPC, GPU fault servicing, or concurrent mutations of one reservation.

CUDA semantics: [NVIDIA VMM introduction](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/) and [CUDA driver API reference](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html).

## Testing Strategy

CUDA unit tests verify page rounding, stable address and preserved contents after growth, zeroed new pages, zero-copy view and DLPack ownership lifetime, reservation bounds, capture rejection, and graph replay after a new page is mapped. The graph test uses a device active count and grows beyond the device's actual granularity. CPU device rejection runs on every build. Tests avoid intentionally accessing unmapped addresses because CUDA faults invalidate the process context.
