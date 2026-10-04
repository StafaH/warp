# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a graph while heterogeneous world component counts change.

This demonstrates ECS allocation and sorting, not MuJoCo physics. Physical
commit happens between replays; logical entity creation/deletion happens inside
one captured graph. Run with ``uv run -m warp.examples.core.example_dynamic_table``.
"""

import argparse
import time

import numpy as np

import warp as wp
from warp.experimental.table import DynamicTable

wp.set_module_options({"enable_backward": False})


@wp.kernel
def create_components(offsets: wp.array[int], qpos: wp.array[float]):
    world = wp.tid()
    begin = offsets[world]
    if begin >= 0:
        for row in range(begin, offsets[world + 1]):
            qpos[row] = float(world) + float(row - begin) * 0.01


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.worlds <= 0 or args.iterations <= 0:
        parser.error("worlds and iterations must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("This captured graph example requires a CUDA device")
    table = DynamicTable(
        {"qpos": float},
        max_rows=args.worlds * 64,
        num_worlds=args.worlds,
        initial_capacity=1,
        device=device,
        dispatch_width=4096,
    )
    counts = wp.zeros(args.worlds, dtype=int, device=device)
    # Warm all kernel specializations and scan scratch with an empty table.
    table.clear()
    offsets = table.append_worlds(counts)
    wp.launch(create_components, dim=args.worlds, inputs=[offsets, table.columns["qpos"]], device=device)
    table.compact()
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as capture:
        table.clear()
        offsets = table.append_worlds(counts)
        wp.launch(create_components, dim=args.worlds, inputs=[offsets, table.columns["qpos"]], device=device)
        table.compact()
    ptr = table.columns["qpos"].ptr
    world = np.arange(args.worlds)
    scenarios = {
        "single robot": np.full(args.worlds, 7, dtype=np.int32),
        "mixed robots": np.asarray([14, 21, 7], dtype=np.int32)[world % 3],
    }
    for name, dof_counts in scenarios.items():
        table.ensure_capacity(int(dof_counts.sum()))
        wp.copy(counts, wp.array(dof_counts, dtype=int, device=device))
        wp.capture_launch(capture.graph)
        table.check_errors()
        np.testing.assert_array_equal(np.diff(table.world_offsets.numpy()), dof_counts)
        if table.columns["qpos"].ptr != ptr:
            raise RuntimeError("Exported qpos pointer moved during table growth")
        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ms = (time.perf_counter() - start) * 1000 / args.iterations
        print(
            f"{name}: worlds={args.worlds}, live rows={table.live_count}, "
            f"committed table/scratch={table.committed_bytes / 2**20:.1f} MiB, "
            f"creation/deletion/grouping graph={ms:.3f} ms"
        )
    print("The same graph and exported qpos pointer served both layouts.")


if __name__ == "__main__":
    main()
