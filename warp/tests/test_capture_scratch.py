# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices


@wp.kernel
def accumulate_last(values: wp.array[wp.int32], output: wp.array[wp.int32]):
    output[0] += values[values.shape[0] - 1]


def test_scratch_conditional_scan_lifetime(test, device):
    with wp.ScopedDevice(device):
        condition = wp.array([1], dtype=wp.int32)
        output = wp.zeros(1, dtype=wp.int32)
        storage = wp.empty(1 << 20, dtype=wp.uint8)
        arena = wp.ScopedCaptureScratch(storage)
        test.assertEqual(arena.memory_kind, storage.memory_kind)
        owner = weakref.ref(arena)
        backing = weakref.ref(storage)

        def body():
            values = wp.full(8, value=1, dtype=wp.int32)
            scanned = wp.empty(8, dtype=wp.int32)
            wp.utils.array_scan(values, scanned, inclusive=True)
            wp.launch(accumulate_last, dim=1, inputs=[scanned, output])

        with arena:
            with wp.ScopedCapture(device=device) as capture:
                wp.capture_if(condition, on_true=body)
        test.assertGreater(arena.used_bytes, 64)  # Native CUB scan scratch also uses the arena.
        del arena, storage
        gc.collect()
        test.assertIsNotNone(owner())
        test.assertIsNotNone(backing())
        for enabled in [0, 1, 1, 0, 1]:
            condition.fill_(enabled)
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), [24])
        del capture
        gc.collect()
        test.assertIsNone(owner())
        test.assertIsNone(backing())


def test_scratch_rewind_branches(test, device):
    with wp.ScopedDevice(device):
        condition = wp.array([1], dtype=wp.int32)
        output = wp.zeros(1, dtype=wp.int32)
        storage = wp.empty(1 << 20, dtype=wp.uint8)
        arena = wp.ScopedCaptureScratch(storage)
        previous_allocator = wp.get_device_allocator(device)
        addresses = []

        def body(value):
            arena.rewind()
            values = wp.full(8, value=value, dtype=wp.int32)
            addresses.append(values.ptr)
            wp.launch(accumulate_last, dim=1, inputs=[values, output])

        with arena:
            with wp.ScopedCapture(device=device) as capture:
                wp.capture_if(condition, on_true=lambda: body(3), on_false=lambda: body(5))
        test.assertIs(wp.get_device_allocator(device), previous_allocator)
        test.assertEqual(addresses[0], addresses[1])
        for enabled in [1, 0, 1]:
            condition.fill_(enabled)
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), [11])
        with test.assertRaisesRegex(RuntimeError, "capture"):
            arena.rewind()


def test_scratch_escape_array_lifetime(test, device):
    with wp.ScopedDevice(device):
        storage = wp.empty(1 << 20, dtype=wp.uint8)
        arena = wp.ScopedCaptureScratch(storage)
        owner = weakref.ref(arena)
        with arena:
            with wp.ScopedCapture(device=device) as capture:
                values = wp.full(8, value=7, dtype=wp.int32)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(values.numpy(), np.full(8, 7, dtype=np.int32))
        del capture, arena, storage
        gc.collect()
        test.assertIsNotNone(owner())
        del values
        gc.collect()
        test.assertIsNone(owner())


def test_scratch_child_graph_lifetime(test, device):
    with wp.ScopedDevice(device):
        storage = wp.empty(1 << 20, dtype=wp.uint8)
        arena = wp.ScopedCaptureScratch(storage)
        owner = weakref.ref(arena)
        condition = wp.ones(1, dtype=wp.int32)
        output = wp.zeros(1, dtype=wp.int32)
        with arena:
            with wp.ScopedCapture(device=device) as child:
                values = wp.full(8, value=7, dtype=wp.int32)
                wp.launch(accumulate_last, dim=1, inputs=[values, output])
        del values
        with wp.ScopedCapture(device=device) as parent:
            wp.capture_if(condition, on_true=child.graph)
        del child, arena, storage
        gc.collect()
        test.assertIsNotNone(owner())
        wp.capture_launch(parent.graph)
        np.testing.assert_array_equal(output.numpy(), [7])
        del parent
        gc.collect()
        test.assertIsNone(owner())


def test_scratch_exhaustion_remains_latched(test, device):
    with wp.ScopedDevice(device):
        arena = wp.ScopedCaptureScratch(wp.empty(256, dtype=wp.uint8))
        previous_allocator = wp.get_device_allocator(device)
        core = wp._src.context.runtime.core
        error_output = core.wp_is_error_output_enabled()
        core.wp_set_error_output_enabled(False)
        try:
            with test.assertRaisesRegex(RuntimeError, "exhausted"):
                with arena:
                    with wp.ScopedCapture(device=device):
                        with test.assertRaisesRegex(RuntimeError, "exhausted"):
                            wp.empty(65, dtype=wp.int32)
                        arena.rewind()
        finally:
            core.wp_set_error_output_enabled(error_output)
        test.assertIs(wp.get_device_allocator(device), previous_allocator)


class TestCaptureScratch(unittest.TestCase):
    def test_cpu_buffer_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            wp.ScopedCaptureScratch(wp.empty(256, dtype=wp.uint8, device="cpu"))


for func in [
    test_scratch_conditional_scan_lifetime,
    test_scratch_rewind_branches,
    test_scratch_escape_array_lifetime,
    test_scratch_child_graph_lifetime,
    test_scratch_exhaustion_remains_latched,
]:
    add_function_test(TestCaptureScratch, func.__name__, func, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
