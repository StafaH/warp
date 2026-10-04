# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices


@wp.kernel
def increment_active(values: wp.array[wp.int32], count: wp.array[wp.int32]):
    i = wp.tid()
    if i < count[0]:
        values[i] += 1


def test_virtual_memory_growth(test, device):
    memory = wp.VirtualMemory(16 * 1024 * 1024, device=device, initial_size=4)
    page = memory.granularity
    test.assertEqual(memory.committed_size, page)
    test.assertGreater(memory.reserved_size, memory.committed_size)
    ptr = memory.ptr
    old = memory.array(1, dtype=wp.int32)
    old.fill_(17)
    memory.commit(page + 4)
    test.assertEqual(memory.ptr, ptr)
    test.assertEqual(old.numpy()[0], 17)
    test.assertEqual(memory.committed_size, 2 * page)
    added = memory.array(page // 4, dtype=wp.int32, offset=page)
    np.testing.assert_array_equal(added.numpy(), np.zeros(page // 4, dtype=np.int32))
    memory.commit(4)
    test.assertEqual(memory.committed_size, 2 * page)
    del old, added
    memory.close()
    memory.close()
    with test.assertRaisesRegex(RuntimeError, "closed"):
        memory.commit(4)


def test_virtual_memory_graph_growth(test, device):
    memory = wp.VirtualMemory(16 * 1024 * 1024, device=device, initial_size=4)
    rows = memory.granularity // 4 + 1
    values = memory.array(rows, dtype=wp.int32)
    count = wp.array([1], dtype=wp.int32, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(increment_active, dim=rows, inputs=[values, count], device=device)
    graph = capture.graph
    wp.capture_launch(graph)
    np.testing.assert_array_equal(memory.array(1, dtype=wp.int32).numpy(), [1])
    memory.commit(rows * 4)
    count.fill_(rows)
    wp.capture_launch(graph)
    expected = np.ones(rows, dtype=np.int32)
    expected[0] = 2
    np.testing.assert_array_equal(values.numpy(), expected)
    test.assertEqual(values.ptr, memory.ptr)
    # Keep the arrays alive while a graph uses their pointers.
    del graph, capture, values
    memory.close()


def test_virtual_memory_validation(test, device):
    memory = wp.VirtualMemory(16 * 1024 * 1024, device=device)
    for size in [-1, memory.reserved_size + 1]:
        with test.assertRaises(ValueError):
            memory.commit(size)
    for shape, offset in [(1, -4), (1, 1), (memory.reserved_size, 0)]:
        with test.assertRaises(ValueError):
            memory.array(shape, dtype=wp.int32, offset=offset)
    view = memory.array(1, dtype=wp.int32)
    with test.assertRaisesRegex(RuntimeError, "views"):
        memory.close()
    del view
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "capture"):
            memory.commit(4)
        with test.assertRaisesRegex(RuntimeError, "capture"):
            memory.close()
        with test.assertRaisesRegex(RuntimeError, "capture"):
            wp.VirtualMemory(4, device=device)
    memory.close()


def test_virtual_memory_view_lifetime(test, device):
    memory = wp.VirtualMemory(4, device=device, initial_size=4)
    reference = weakref.ref(memory)
    view = memory.array(1, dtype=wp.int32)
    del memory
    gc.collect()
    test.assertIsNotNone(reference())
    view.fill_(9)
    np.testing.assert_array_equal(view.numpy(), [9])
    del view
    gc.collect()
    test.assertIsNone(reference())


def test_virtual_memory_dlpack_lifetime(test, device):
    memory = wp.VirtualMemory(16, device=device, initial_size=16)
    reference = weakref.ref(memory)
    view = memory.array(4, dtype=wp.int32)
    alias = wp.from_dlpack(wp.to_dlpack(view))
    test.assertEqual(alias.ptr, view.ptr)
    alias.fill_(23)
    np.testing.assert_array_equal(view.numpy(), [23, 23, 23, 23])
    with test.assertRaisesRegex(RuntimeError, "views"):
        memory.close()
    del memory, view
    gc.collect()
    test.assertIsNotNone(reference())
    np.testing.assert_array_equal(alias.numpy(), [23, 23, 23, 23])
    del alias
    gc.collect()
    test.assertIsNone(reference())


class TestVirtualMemory(unittest.TestCase):
    def test_cpu_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            wp.VirtualMemory(4, device="cpu")


for func in [
    test_virtual_memory_growth,
    test_virtual_memory_graph_growth,
    test_virtual_memory_validation,
    test_virtual_memory_view_lifetime,
    test_virtual_memory_dlpack_lifetime,
]:
    add_function_test(TestVirtualMemory, func.__name__, func, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
