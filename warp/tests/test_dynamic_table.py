# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.experimental.table import DynamicTable, Entity, TableState, append_row, delete_entity, entity_at, find_row

wp.set_module_options({"enable_backward": False})


@wp.kernel
def append_entities(
    state: TableState,
    worlds: wp.array[int],
    values: wp.array[int],
    component: wp.array[int],
    vectors: wp.array[wp.vec3],
    handles: wp.array[Entity],
    rows: wp.array[int],
):
    i = wp.tid()
    row = append_row(state, worlds[i])
    rows[i] = row
    if row >= 0:
        component[row] = values[i]
        vectors[row] = wp.vec3(float(values[i]), 2.0, 3.0)
        handles[i] = entity_at(state, row)


@wp.kernel
def erase_entities(state: TableState, handles: wp.array[Entity]):
    delete_entity(state, handles[wp.tid()])


@wp.kernel
def resolve_entities(state: TableState, handles: wp.array[Entity], rows: wp.array[int]):
    i = wp.tid()
    rows[i] = find_row(state, handles[i])


class TestDynamicTable(unittest.TestCase):
    def devices(self):
        return [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def append(self, table, worlds, values):
        count = len(worlds)
        handles = wp.empty(count, dtype=Entity, device=table.device)
        rows = wp.empty(count, dtype=int, device=table.device)
        wp.launch(
            append_entities,
            dim=count,
            inputs=[
                table.state,
                wp.array(worlds, dtype=int, device=table.device),
                wp.array(values, dtype=int, device=table.device),
                table.columns["value"],
                table.columns["vector"],
                handles,
                rows,
            ],
            device=table.device,
        )
        return handles, rows

    def test_overflow_and_invalid_world(self):
        for device in self.devices():
            with self.subTest(device=device):
                table = DynamicTable({"value": int, "vector": wp.vec3}, 100, 3, initial_capacity=2, device=device)
                _, rows = self.append(table, [0] * 100, list(range(100)))
                self.assertEqual(np.count_nonzero(rows.numpy() >= 0), 2)
                self.assertEqual(table.live_count, 2)
                self.assertEqual(int(table.state.counters.numpy()[0]), 2)
                with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                    table.check_errors()
                table.clear_errors()
                _, rows = self.append(table, [-1, 3], [7, 8])
                np.testing.assert_array_equal(rows.numpy(), [-1, -1])
                with self.assertRaisesRegex(ValueError, "invalid world"):
                    table.check_errors()
                self.assertEqual(table.live_count, 2)

    def test_compaction_handles_and_recycling(self):
        for device in self.devices():
            with self.subTest(device=device):
                table = DynamicTable(
                    {"value": int, "vector": wp.vec3}, 32, 3, initial_capacity=4, device=device, dispatch_width=2
                )
                handles, resolved = self.append(table, [2, 0, 1, 0], [10, 11, 12, 13])
                ptr = table.columns["value"].ptr
                # Delete the same handle twice; only the first deletion changes counts/generation.
                wp.launch(erase_entities, dim=1, inputs=[table.state, handles], device=device)
                wp.launch(erase_entities, dim=1, inputs=[table.state, handles], device=device)
                self.assertEqual(table.live_count, 3)
                table.compact()
                np.testing.assert_array_equal(table.world_offsets.numpy(), [0, 2, 3, 3])
                np.testing.assert_array_equal(table.state.world.numpy()[:3], [0, 0, 1])
                wp.launch(resolve_entities, dim=4, inputs=[table.state, handles, resolved], device=device)
                rows = resolved.numpy()
                self.assertEqual(rows[0], -1)
                np.testing.assert_array_equal(table.columns["value"].numpy()[rows[1:]], [11, 12, 13])
                np.testing.assert_array_equal(table.columns["vector"].numpy()[rows[1:], 0], [11, 12, 13])
                fresh, _ = self.append(table, [2], [99])
                table.check_errors()
                self.assertEqual(table.live_count, 4)
                table.compact()
                self.assertEqual(table.columns["value"].ptr, ptr)
                wp.launch(resolve_entities, dim=4, inputs=[table.state, handles, resolved], device=device)
                self.assertEqual(resolved.numpy()[0], -1)
                # The recycled slot has the same ID but a different generation.
                old = handles.numpy()[0]
                new = fresh.numpy()[0]
                self.assertEqual(old["slot"], new["slot"])
                self.assertNotEqual(old["generation"], new["generation"])
                np.testing.assert_array_equal(table.world_offsets.numpy(), [0, 2, 3, 4])

    def test_graph_growth_preserves_addresses_and_data(self):
        for device in wp.get_cuda_devices():
            with self.subTest(device=device):
                table = DynamicTable(
                    {"value": int, "vector": wp.vec3}, 2000000, 1, initial_capacity=1, device=device, dispatch_width=256
                )
                worlds = wp.array([-1], dtype=int, device=device)
                values = wp.array([10], dtype=int, device=device)
                handles = wp.empty(1, dtype=Entity, device=device)
                rows = wp.empty(1, dtype=int, device=device)
                args = [table.state, worlds, values, table.columns["value"], table.columns["vector"], handles, rows]
                # Warm specialization and native scan scratch before capture without creating a row.
                wp.launch(append_entities, dim=1, inputs=args, device=device)
                table.clear_errors()
                table.compact()
                wp.copy(worlds, wp.array([0], dtype=int, device=device))
                wp.synchronize_device(device)
                with wp.ScopedCapture(device=device) as capture:
                    wp.launch(append_entities, dim=1, inputs=args, device=device)
                    table.compact()
                pointers = {name: array.ptr for name, array in table.columns.items()}
                before = table.committed_bytes
                wp.capture_launch(capture.graph)
                self.assertEqual(table.live_count, 1)
                table.ensure_capacity(table._regions[0][0].granularity // 4 + 1)
                self.assertGreater(table.committed_bytes, before)
                wp.copy(values, wp.array([20], dtype=int, device=device))
                wp.capture_launch(capture.graph)
                table.check_errors()
                self.assertEqual(table.live_count, 2)
                np.testing.assert_array_equal(np.sort(table.columns["value"][:2].numpy()), [10, 20])
                self.assertEqual(pointers, {name: array.ptr for name, array in table.columns.items()})
                np.testing.assert_array_equal(table.world_offsets.numpy(), [0, 2])

    def test_bulk_append_and_clear(self):
        for device in self.devices():
            with self.subTest(device=device):
                table = DynamicTable({"value": int, "vector": wp.vec3}, 8, 3, initial_capacity=8, device=device)
                handles, rows = self.append(table, [1, 2], [10, 11])
                table.clear()
                offsets = table.append_worlds(wp.array([2, 0, 3], dtype=int, device=device))
                np.testing.assert_array_equal(offsets.numpy(), [0, 2, 2, 5])
                self.assertEqual(table.live_count, 5)
                np.testing.assert_array_equal(table.state.world.numpy()[:5], [0, 0, 2, 2, 2])
                wp.launch(resolve_entities, dim=2, inputs=[table.state, handles, rows], device=device)
                np.testing.assert_array_equal(rows.numpy(), [-1, -1])
                # The sum exceeds int32; the wide scan must reject the whole batch safely.
                offsets = table.append_worlds(wp.array([2**31 - 1, 2**31 - 1, 0], dtype=int, device=device))
                np.testing.assert_array_equal(offsets.numpy(), [-1] * 4)
                self.assertEqual(table.live_count, 5)
                with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                    table.check_errors()
                table.clear_errors()
                offsets = table.append_worlds(wp.array([1, -1, 1], dtype=int, device=device))
                np.testing.assert_array_equal(offsets.numpy(), [-1] * 4)
                self.assertEqual(table.live_count, 5)
                with self.assertRaisesRegex(ValueError, "negative row count"):
                    table.check_errors()
                table.clear_errors()
                table.clear()
                table.check_errors()
                self.assertEqual(table.live_count, 0)
                np.testing.assert_array_equal(table.world_offsets.numpy(), [0] * 4)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
