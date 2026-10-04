# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-only spawning and world resets with persistent particles in a DynamicTable.

One graph changes populations, deletes entities, compacts, spawns, and integrates particles.
All physical capacity is committed at startup. Readback occurs after the 32-step rollout.
Run: uv run -m warp.examples.core.example_dynamic_particle_worlds --worlds 8192
"""

import argparse
import json

import numpy as np

import warp as wp
from warp.experimental.table import DynamicTable, Entity, TableState, delete_entity, entity_at, find_row

wp.set_module_options({"enable_backward": False})


@wp.kernel
def plan_population(tick: wp.array[int], reset: wp.array[int], counts: wp.array[int]):
    world = wp.tid()
    step = tick[0]
    reset[world] = 0
    counts[world] = 0
    if step == 0:
        reset[world] = 1
        counts[world] = 4
    elif step == 8 and world % 2 == 0:
        reset[world] = 1
        counts[world] = 12
    elif step == 16:
        reset[world] = 1
        counts[world] = 1
    elif step == 24:
        reset[world] = 1
        counts[world] = (world % 4 + 1) * 3


@wp.kernel
def erase_selected(state: TableState, reset: wp.array[int], width: int):
    for row in range(wp.tid(), state.counters[0], width):
        if state.alive[row] != 0 and reset[state.world[row]] != 0:
            delete_entity(state, entity_at(state, row))


@wp.kernel
def initialize_particles(
    state: TableState,
    tick: wp.array[int],
    offsets: wp.array[int],
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    age: wp.array[float],
    original: wp.array[Entity],
):
    world = wp.tid()
    begin = offsets[world]
    if begin >= 0:
        for row in range(begin, offsets[world + 1]):
            local = row - begin
            position[row] = wp.vec3(float(local) * 0.1, 0.0, 1.0)
            velocity[row] = wp.vec3(0.1, 0.05 * float(world % 3), 0.0)
            age[row] = 0.0
            if tick[0] == 0 and local == 0:
                original[world] = entity_at(state, row)


@wp.kernel
def integrate(
    state: TableState, position: wp.array[wp.vec3], velocity: wp.array[wp.vec3], age: wp.array[float], width: int
):
    for row in range(wp.tid(), state.counters[0], width):
        if state.alive[row] != 0:
            velocity[row] += wp.vec3(0.0, 0.0, -9.81) * 0.01
            position[row] += velocity[row] * 0.01
            age[row] += 0.01


@wp.kernel
def validate_and_record(
    state: TableState,
    tick: wp.array[int],
    offsets: wp.array[int],
    original: wp.array[Entity],
    age: wp.array[float],
    history: wp.array2d[int],
    errors: wp.array[int],
):
    world = wp.tid()
    step = tick[0]
    row = find_row(state, original[world])
    survives = step < 8 or (step < 16 and world % 2 == 1)
    if (row >= 0) != survives:
        wp.atomic_or(errors, 0, 1)
    if row >= 0:
        if state.world[row] != world:
            wp.atomic_or(errors, 0, 2)
        if wp.abs(age[row] - float(step + 1) * 0.01) > 0.00001:
            wp.atomic_or(errors, 0, 4)
    if step % 8 == 0:
        history[step / 8, world] = offsets[world + 1] - offsets[world]


@wp.kernel
def advance_tick(tick: wp.array[int]):
    tick[0] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 < args.worlds < 2**27:
        parser.error("worlds must be positive and its particle capacity must fit int32")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("CUDA graph capture requires a CUDA device")
    table = DynamicTable(
        {"position": wp.vec3, "velocity": wp.vec3, "age": float},
        max_rows=args.worlds * 16,
        num_worlds=args.worlds,
        initial_capacity=args.worlds * 16,
        dispatch_width=4096,
        device=device,
    )
    with wp.ScopedDevice(device):
        tick = wp.zeros(1, dtype=int)
        reset = wp.zeros(args.worlds, dtype=int)
        counts = wp.zeros(args.worlds, dtype=int)
        original = wp.zeros(args.worlds, dtype=Entity)
        history = wp.zeros((4, args.worlds), dtype=int)
        errors = wp.zeros(1, dtype=int)
        position, velocity, age = (table.columns[name] for name in ("position", "velocity", "age"))

        def timestep():
            wp.launch(plan_population, dim=args.worlds, inputs=[tick, reset, counts])
            wp.launch(erase_selected, dim=table.dispatch_width, inputs=[table.state, reset, table.dispatch_width])
            table.compact()  # Reclaim deleted rows before reserving new entities.
            offsets = table.append_worlds(counts)
            wp.launch(
                initialize_particles,
                dim=args.worlds,
                inputs=[table.state, tick, offsets, position, velocity, age, original],
            )
            table.compact()  # Publish grouped world ranges after the append.
            wp.launch(
                integrate, dim=table.dispatch_width, inputs=[table.state, position, velocity, age, table.dispatch_width]
            )
            wp.launch(
                validate_and_record,
                dim=args.worlds,
                inputs=[table.state, tick, table.world_offsets, original, age, history, errors],
            )
            wp.launch(advance_tick, dim=1, inputs=[tick])

        timestep()  # Setup warmup: compile specializations and initialize scan workspace.
        wp.synchronize_device(device)
        table.clear()
        table.clear_errors()
        tick.zero_()
        errors.zero_()
        history.zero_()
        with wp.ScopedCapture(device=device) as capture:
            timestep()
        graph = capture.graph
        pointers = [column.ptr for column in table.columns.values()]
        committed = table.committed_bytes
        for _ in range(32):
            wp.capture_launch(graph)

        # All host inspection is after the GPU-driven rollout.
        table.check_errors()
        np.testing.assert_array_equal(errors.numpy(), 0)
        world = np.arange(args.worlds)
        expected = np.stack(
            [
                np.full(args.worlds, 4),
                np.where(world % 2 == 0, 12, 4),
                np.ones(args.worlds, dtype=int),
                (world % 4 + 1) * 3,
            ]
        )
        np.testing.assert_array_equal(history.numpy(), expected)
        np.testing.assert_array_equal(np.diff(table.world_offsets.numpy()), expected[-1])
        if [column.ptr for column in table.columns.values()] != pointers:
            raise RuntimeError("Component column addresses moved during replay")
        if table.committed_bytes != committed:
            raise RuntimeError("Physical table backing changed during replay")
        if not np.all(np.isfinite(position[: table.live_count].numpy())):
            raise RuntimeError("Particle integration produced nonfinite positions")
        print(
            json.dumps(
                {
                    "worlds": args.worlds,
                    "steps": int(tick.numpy()[0]),
                    "particle_counts_at_steps_0_8_16_24": expected.sum(axis=1).tolist(),
                    "table_owned_MiB": committed / 2**20,
                    "graph_and_column_addresses_reused": True,
                    "surviving_handles_and_episode_ages_verified": True,
                    "deleted_handles_invalidated": True,
                    "physical_growth_during_rollout": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
