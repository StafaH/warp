# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render pure-Warp variable sphere populations and packed contact tables offline.

Compare two captured GPU simulations: fixed per-world particle/contact buffers and
globally shared DynamicTables. The HTML plays recorded GPU frames; it runs no physics.
Both allocate physical storage at startup and make all reset decisions on the GPU.
Run: uv run -m warp.examples.core.example_dynamic_collision_worlds --output collision_worlds.html
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import warp as wp
from warp.experimental.table import DynamicTable, TableState, delete_entity, entity_at

wp.set_module_options({"enable_backward": False})

MAX_PARTICLES = 512
MAX_CONTACTS = 2048
STEPS = 256
SAMPLE_INTERVAL = 4
RADIUS = 0.035


@wp.kernel
def plan(tick: wp.array[int], reset: wp.array[int], spawn: wp.array[int], population: wp.array[int]):
    world = wp.tid()
    step = tick[0]
    reset[world] = 0
    spawn[world] = 0
    if step == 0:
        reset[world] = 1
        spawn[world] = 4
    elif step == 64 and world % 64 == 0:
        reset[world] = 1
        spawn[world] = 512
    elif step == 128:
        reset[world] = 1
        spawn[world] = 1
    elif step == 192:
        reset[world] = 1
        spawn[world] = (world % 4 + 1) * 3
    if reset[world] != 0:
        population[world] = spawn[world]


@wp.kernel
def erase(state: TableState, reset: wp.array[int]):
    row = wp.tid()
    if row < state.counters[0] and state.alive[row] != 0:
        if reset[state.world[row]] != 0:
            delete_entity(state, entity_at(state, row))


@wp.kernel
def initialize(
    reset: wp.array[int],
    spawn: wp.array[int],
    offsets: wp.array[int],
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    age: wp.array[float],
    local_id: wp.array[int],
):
    world = wp.tid()
    if reset[world] != 0 and offsets[world] >= 0:
        for local in range(spawn[world]):
            row = offsets[world] + local
            x = 0.25 + float(local % 16) * 0.075
            z = 0.15 + float(local / 16) * 0.065
            position[row] = wp.vec3(x, 0.0, z)
            velocity[row] = wp.vec3(0.1 * float(world % 3 - 1), 0.0, 0.0)
            age[row] = 0.0
            local_id[row] = local


@wp.func
def world_for_row(mode: int, row: int, ownership: wp.array[int]) -> int:
    world = row / 512
    if mode == 1:
        world = ownership[row]
    return world


@wp.kernel
def advance(
    mode: int,
    extent: wp.array[int],
    ownership: wp.array[int],
    offsets: wp.array[int],
    population: wp.array[int],
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    age: wp.array[float],
):
    row = wp.tid()
    if row >= extent[0]:
        return
    world = world_for_row(mode, row, ownership)
    if row - offsets[world] >= population[world]:
        return
    v = velocity[row] + wp.vec3(0.0, 0.0, -9.81) * 0.005
    p = position[row] + v * 0.005
    if p[2] < 0.035:
        p = wp.vec3(p[0], 0.0, 0.035)
        v = wp.vec3(v[0] * 0.995, 0.0, wp.abs(v[2]) * 0.2)
    if p[0] < 0.035 or p[0] > 2.965:
        p = wp.vec3(wp.clamp(p[0], 0.035, 2.965), 0.0, p[2])
        v = wp.vec3(-v[0] * 0.2, 0.0, v[2])
    position[row] = p
    velocity[row] = v
    age[row] += 0.005


@wp.kernel
def count_pairs(
    extent: wp.array[int],
    ownership: wp.array[int],
    offsets: wp.array[int],
    population: wp.array[int],
    position: wp.array[wp.vec3],
    counts: wp.array[int],
):
    row = wp.tid()
    if row >= extent[0]:
        return
    world = ownership[row]
    for other in range(row + 1, offsets[world] + population[world]):
        if wp.length_sq(position[other] - position[row]) < 0.0049:
            wp.atomic_add(counts, world, 1)


@wp.kernel
def write_pairs(
    mode: int,
    extent: wp.array[int],
    ownership: wp.array[int],
    offsets: wp.array[int],
    population: wp.array[int],
    position: wp.array[wp.vec3],
    contact_offsets: wp.array[int],
    cursor: wp.array[int],
    a: wp.array[int],
    b: wp.array[int],
    normal: wp.array[wp.vec3],
    depth: wp.array[float],
    errors: wp.array[int],
):
    row = wp.tid()
    if row >= extent[0]:
        return
    world = world_for_row(mode, row, ownership)
    if row - offsets[world] >= population[world] or contact_offsets[world] < 0:
        return
    for other in range(row + 1, offsets[world] + population[world]):
        delta = position[other] - position[row]
        distance_sq = wp.length_sq(delta)
        if distance_sq < 0.0049:
            local = wp.atomic_add(cursor, world, 1)
            if mode == 0 and local >= 2048:
                wp.atomic_or(errors, 0, 1)
                continue
            contact = contact_offsets[world] + local
            distance = wp.sqrt(distance_sq)
            n = wp.vec3(1.0, 0.0, 0.0)
            if distance > 0.000001:
                n = delta / distance
            a[contact] = row
            b[contact] = other
            normal[contact] = n
            depth[contact] = 0.07 - distance


@wp.kernel
def solve_contacts(
    mode: int,
    extent: wp.array[int],
    ownership: wp.array[int],
    offsets: wp.array[int],
    counts: wp.array[int],
    a: wp.array[int],
    b: wp.array[int],
    normal: wp.array[wp.vec3],
    depth: wp.array[float],
    velocity: wp.array[wp.vec3],
    correction: wp.array[wp.vec3d],
    impulse: wp.array[wp.vec3d],
):
    contact = wp.tid()
    if contact >= extent[0]:
        return
    world = contact / 2048
    if mode == 1:
        world = ownership[contact]
    if contact - offsets[world] >= wp.min(counts[world], 2048):
        return
    first, second = a[contact], b[contact]
    n = normal[contact]
    c = n * depth[contact] * 0.45
    cd = wp.vec3d(wp.float64(c[0]), wp.float64(c[1]), wp.float64(c[2]))
    wp.atomic_add(correction, first, -cd)
    wp.atomic_add(correction, second, cd)
    relative = wp.dot(velocity[second] - velocity[first], n)
    if relative < 0.0:
        v = n * (-0.55 * relative)
        vd = wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))
        wp.atomic_add(impulse, first, -vd)
        wp.atomic_add(impulse, second, vd)


@wp.kernel
def apply_corrections(
    mode: int,
    extent: wp.array[int],
    ownership: wp.array[int],
    offsets: wp.array[int],
    population: wp.array[int],
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    correction: wp.array[wp.vec3d],
    impulse: wp.array[wp.vec3d],
):
    row = wp.tid()
    if row >= extent[0]:
        return
    world = world_for_row(mode, row, ownership)
    if row - offsets[world] >= population[world]:
        return
    c, v = correction[row], impulse[row]
    position[row] += wp.vec3(float(c[0]), float(c[1]), float(c[2]))
    velocity[row] += wp.vec3(float(v[0]), float(v[1]), float(v[2]))


@wp.kernel
def record_particles(
    tick: wp.array[int],
    visible: wp.array[int],
    population: wp.array[int],
    offsets: wp.array[int],
    position: wp.array[wp.vec3],
    local_id: wp.array[int],
    history: wp.array3d[wp.vec3],
    counts: wp.array2d[int],
):
    view, local = wp.tid()
    step = tick[0]
    if step % 4 != 0:
        return
    frame, world = step / 4, visible[view]
    if local == 0:
        counts[frame, view] = population[world]
    if local < population[world]:
        row = offsets[world] + local
        history[frame, view, local_id[row]] = position[row]


@wp.kernel
def record_contacts(
    tick: wp.array[int],
    visible: wp.array[int],
    counts: wp.array[int],
    offsets: wp.array[int],
    local_id: wp.array[int],
    a: wp.array[int],
    b: wp.array[int],
    history: wp.array3d[wp.vec2i],
    history_counts: wp.array2d[int],
):
    view, local = wp.tid()
    step = tick[0]
    if step % 4 != 0:
        return
    frame, world = step / 4, visible[view]
    if local == 0:
        history_counts[frame, view] = wp.min(counts[world], 2048)
    if local < counts[world] and local < 2048 and offsets[world] >= 0:
        row = offsets[world] + local
        history[frame, view, local] = wp.vec2i(local_id[a[row]], local_id[b[row]])


@wp.kernel
def record_totals(tick: wp.array[int], population: wp.array[int], contacts: wp.array[int], totals: wp.array2d[int]):
    world = wp.tid()
    if tick[0] % 4 == 0:
        wp.atomic_add(totals, tick[0] / 4, 0, population[world])
        wp.atomic_add(totals, tick[0] / 4, 1, contacts[world])


@wp.kernel
def next_tick(tick: wp.array[int]):
    tick[0] += 1


def allocated(arrays):
    return sum(array.capacity for array in arrays)


def simulate(worlds, visible_ids, packed, device):
    """Capture and execute one backend; read recorded data only after the rollout."""
    mode = int(packed)
    peak = max(4 * worlds + 508 * ((worlds + 63) // 64), sum((w % 4 + 1) * 3 for w in range(worlds)))
    contact_capacity = max(2048, worlds * 32)
    with wp.ScopedDevice(device):
        tick = wp.zeros(1, dtype=int)
        reset = wp.zeros(worlds, dtype=int)
        spawn = wp.zeros(worlds, dtype=int)
        population = wp.zeros(worlds, dtype=int)
        contact_counts = wp.zeros(worlds, dtype=int)
        cursor = wp.zeros(worlds, dtype=int)
        errors = wp.zeros(1, dtype=int)
        common = [tick, reset, spawn, population, contact_counts, cursor, errors]
        tables = []
        if packed:
            particles = DynamicTable(
                {"position": wp.vec3, "velocity": wp.vec3, "age": float, "local_id": int},
                worlds * MAX_PARTICLES,
                worlds,
                initial_capacity=peak,
                device=device,
            )
            contacts = DynamicTable(
                {"a": int, "b": int, "normal": wp.vec3, "depth": float},
                worlds * MAX_CONTACTS,
                worlds,
                initial_capacity=contact_capacity,
                device=device,
            )
            tables = [particles, contacts]
            p = SimpleNamespace(**particles.columns)
            c = SimpleNamespace(**contacts.columns)
            particle_extent, particle_owner = particles.state.counters, particles.state.world
            contact_extent, contact_owner = contacts.state.counters, contacts.state.world
            particle_offsets = particles.world_offsets
            particle_dispatch, contact_dispatch = peak, contact_capacity
        else:
            particle_dispatch = worlds * MAX_PARTICLES
            contact_dispatch = worlds * MAX_CONTACTS
            p = SimpleNamespace(
                position=wp.empty(particle_dispatch, dtype=wp.vec3),
                velocity=wp.empty(particle_dispatch, dtype=wp.vec3),
                age=wp.empty(particle_dispatch),
                local_id=wp.empty(particle_dispatch, dtype=int),
            )
            c = SimpleNamespace(
                a=wp.empty(contact_dispatch, dtype=int),
                b=wp.empty(contact_dispatch, dtype=int),
                normal=wp.empty(contact_dispatch, dtype=wp.vec3),
                depth=wp.empty(contact_dispatch),
            )
            particle_extent = wp.array([particle_dispatch], dtype=int)
            contact_extent = wp.array([contact_dispatch], dtype=int)
            particle_owner = contact_owner = wp.empty(0, dtype=int)
            particle_offsets = wp.array(np.arange(worlds + 1, dtype=np.int32) * MAX_PARTICLES, dtype=int)
            dense_contact_offsets = wp.array(np.arange(worlds + 1, dtype=np.int32) * MAX_CONTACTS, dtype=int)
            common.extend([particle_extent, contact_extent, particle_offsets, dense_contact_offsets])
        correction = wp.zeros(particle_dispatch, dtype=wp.vec3d)
        impulse = wp.zeros(particle_dispatch, dtype=wp.vec3d)
        visible = wp.array(visible_ids, dtype=int)
        views, frames = len(visible_ids), STEPS // SAMPLE_INTERVAL
        positions = wp.zeros((frames, views, MAX_PARTICLES), dtype=wp.vec3)
        recorded_counts = wp.zeros((frames, views), dtype=int)
        pairs = wp.zeros((frames, views, MAX_CONTACTS), dtype=wp.vec2i)
        recorded_contacts = wp.zeros((frames, views), dtype=int)
        totals = wp.zeros((frames, 2), dtype=int)
        recording = [visible, positions, recorded_counts, pairs, recorded_contacts, totals]

        def timestep():
            wp.launch(plan, dim=worlds, inputs=[tick, reset, spawn, population])
            if packed:
                wp.launch(erase, dim=peak, inputs=[particles.state, reset])
                particles.compact()
                new_offsets = particles.append_worlds(spawn)
            else:
                new_offsets = particle_offsets
            wp.launch(
                initialize, dim=worlds, inputs=[reset, spawn, new_offsets, p.position, p.velocity, p.age, p.local_id]
            )
            if packed:
                particles.compact()
            wp.launch(
                advance,
                dim=particle_dispatch,
                inputs=[
                    mode,
                    particle_extent,
                    particle_owner,
                    particle_offsets,
                    population,
                    p.position,
                    p.velocity,
                    p.age,
                ],
            )
            contact_counts.zero_()
            cursor.zero_()
            if packed:
                contacts.clear()
                wp.launch(
                    count_pairs,
                    dim=peak,
                    inputs=[particle_extent, particle_owner, particle_offsets, population, p.position, contact_counts],
                )
                contact_offsets = contacts.append_worlds(contact_counts)
            else:
                contact_offsets = dense_contact_offsets
            wp.launch(
                write_pairs,
                dim=particle_dispatch,
                inputs=[
                    mode,
                    particle_extent,
                    particle_owner,
                    particle_offsets,
                    population,
                    p.position,
                    contact_offsets,
                    cursor,
                    c.a,
                    c.b,
                    c.normal,
                    c.depth,
                    errors,
                ],
            )
            solver_counts = contact_counts if packed else cursor
            correction.zero_()
            impulse.zero_()
            wp.launch(
                solve_contacts,
                dim=contact_dispatch,
                inputs=[
                    mode,
                    contact_extent,
                    contact_owner,
                    contact_offsets,
                    solver_counts,
                    c.a,
                    c.b,
                    c.normal,
                    c.depth,
                    p.velocity,
                    correction,
                    impulse,
                ],
            )
            wp.launch(
                apply_corrections,
                dim=particle_dispatch,
                inputs=[
                    mode,
                    particle_extent,
                    particle_owner,
                    particle_offsets,
                    population,
                    p.position,
                    p.velocity,
                    correction,
                    impulse,
                ],
            )
            wp.launch(
                record_particles,
                dim=(views, MAX_PARTICLES),
                inputs=[
                    tick,
                    visible,
                    population,
                    particle_offsets,
                    p.position,
                    p.local_id,
                    positions,
                    recorded_counts,
                ],
            )
            wp.launch(
                record_contacts,
                dim=(views, MAX_CONTACTS),
                inputs=[tick, visible, solver_counts, contact_offsets, p.local_id, c.a, c.b, pairs, recorded_contacts],
            )
            wp.launch(record_totals, dim=worlds, inputs=[tick, population, solver_counts, totals])
            wp.launch(next_tick, dim=1, inputs=[tick])

        timestep()
        wp.synchronize_device(device)
        for table in tables:
            table.clear()
            table.clear_errors()
        tick.zero_()
        population.zero_()
        errors.zero_()
        totals.zero_()
        with wp.ScopedCapture(device=device) as capture:
            timestep()
        graph = capture.graph
        pointers = [array.ptr for array in [*vars(p).values(), *vars(c).values()]]
        memory = {
            "particle_bytes": particles.committed_bytes + allocated([correction, impulse])
            if packed
            else allocated([*vars(p).values(), correction, impulse]),
            "contact_bytes": contacts.committed_bytes if packed else allocated(vars(c).values()),
            "mapping_bytes": allocated(common),
        }
        memory["bytes"] = sum(memory.values())
        start, end = wp.Event(device=device, enable_timing=True), wp.Event(device=device, enable_timing=True)
        wp.record_event(start)
        for _ in range(STEPS):
            wp.capture_launch(graph)
        wp.record_event(end)
        # Timing and visual readback happen after the whole GPU-driven rollout.
        memory["rollout_ms"] = wp.get_event_elapsed_time(start, end)
        for table in tables:
            table.check_errors()
        np.testing.assert_array_equal(errors.numpy(), 0)
        if [array.ptr for array in [*vars(p).values(), *vars(c).values()]] != pointers:
            raise RuntimeError("Recorded component addresses changed")
        if packed and (
            particles.committed_bytes + allocated([correction, impulse]) != memory["particle_bytes"]
            or contacts.committed_bytes != memory["contact_bytes"]
        ):
            raise RuntimeError("Physical backing changed during replay")
        result = {
            "positions": positions.numpy(),
            "counts": recorded_counts.numpy(),
            "pairs": pairs.numpy(),
            "contacts": recorded_contacts.numpy(),
            "totals": totals.numpy(),
            "memory": memory,
            "recording_bytes": allocated(recording),
            "particle_capacity": peak,
            "contact_capacity": contact_capacity,
        }
        for frame in range(frames):
            for view in range(views):
                values = result["positions"][frame, view, : result["counts"][frame, view]]
                if not np.all(np.isfinite(values)):
                    raise RuntimeError("Recorded simulation contains nonfinite positions")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("dynamic_collision_worlds.html"))
    args = parser.parse_args()
    if not 0 < args.worlds < 2**20:
        parser.error("worlds must be positive and contact indexing must fit int32")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("The captured demo requires CUDA")
    visible = [w for w in (0, 1, 2, 64, 65, 128, 129, 192) if w < args.worlds]
    dense = simulate(args.worlds, visible, False, device)
    packed = simulate(args.worlds, visible, True, device)
    np.testing.assert_array_equal(packed["counts"], dense["counts"])
    max_error = 0.0
    for frame in range(STEPS // SAMPLE_INTERVAL):
        for view in range(len(visible)):
            count = packed["counts"][frame, view]
            first, second = packed["positions"][frame, view, :count], dense["positions"][frame, view, :count]
            np.testing.assert_allclose(first, second, atol=0.003, rtol=0.003)
            if count:
                max_error = max(max_error, float(np.max(np.abs(first - second))))
    world_ids = np.arange(args.worlds)
    expected = [
        4 * args.worlds,
        4 * args.worlds + 508 * ((args.worlds + 63) // 64),
        args.worlds,
        int(((world_ids % 4 + 1) * 3).sum()),
    ]
    np.testing.assert_array_equal(packed["totals"][[0, 16, 32, 48], 0], expected)
    if not np.any(packed["totals"][:, 1] > 0):
        raise RuntimeError("The example did not generate sphere contacts")
    phases = ["Sparse population", "Burst in one of every 64 worlds", "Reset to one sphere", "Mixed populations"]
    frames = []
    for frame in range(STEPS // SAMPLE_INTERVAL):
        views = []
        for index, world in enumerate(visible):
            count = int(packed["counts"][frame, index])
            pairs = packed["pairs"][frame, index, : packed["contacts"][frame, index]]
            if np.any(pairs < 0) or np.any(pairs >= count):
                raise RuntimeError("Contact references an invalid local particle")
            views.append(
                {
                    "id": world,
                    "particles": np.round(packed["positions"][frame, index, :count][:, [0, 2]], 5).tolist(),
                    "contacts": pairs.tolist(),
                }
            )
        frames.append(
            {
                "phase": phases[frame // 16],
                "total_particles": int(packed["totals"][frame, 0]),
                "total_contacts": int(packed["totals"][frame, 1]),
                "worlds": views,
            }
        )
    data = {
        "worlds": args.worlds,
        "steps": STEPS,
        "visible_worlds": visible,
        "frames": frames,
        "comparison": {
            "dense": dense["memory"],
            "packed": packed["memory"],
            "common_recording_bytes": packed["recording_bytes"],
            "particle_global_capacity": packed["particle_capacity"],
            "contact_global_capacity": packed["contact_capacity"],
            "max_particles_per_world": MAX_PARTICLES,
            "max_contacts_per_world": MAX_CONTACTS,
            "validated": True,
            "max_position_difference": max_error,
        },
        "hardware": device.name,
        "notes": [
            "Physics, resets, collision detection, contact generation and response are implemented in Warp kernels.",
            "The browser plays GPU-recorded frames; it performs no simulation. Readback happens after rollout.",
            "Packed backing is committed once for this sparse-burst scenario. Allocated memory does not shrink during replay.",
            "A simultaneous 512-sphere burst in every world exceeds the chosen packed particle budget; a GPU append fails.",
            "The dense path supports 512 particles and 2048 contacts independently in every world at a larger memory cost.",
            "Both paths use brute-force within-world pair tests and one simple contact-response pass, not a production solver.",
            "Double-precision correction accumulators reduce sensitivity to contact insertion order; state is float32.",
            "Memory includes component buffers, solver accumulators, metadata and packed compaction scratch; graph/JIT/cache overhead is excluded.",
            "Common GPU frame-recording buffers are excluded from backend totals and reported separately; both backends allocate them.",
            "Rollout times include GPU frame recording; one timed rollout follows one-step compilation warmup. They are illustrative, not steady-state throughput benchmarks.",
        ],
    }
    template = Path(__file__).parent / "assets/dynamic_collision_worlds.html"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.read_text().replace("__DEMO_DATA__", json.dumps(data, separators=(",", ":"))))
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "dense_bytes": dense["memory"]["bytes"],
                "packed_bytes": packed["memory"]["bytes"],
                "population_stages": expected,
                "max_contacts": int(packed["totals"][:, 1].max()),
                "max_position_difference": max_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
