import numpy as np
import warp as wp

from warp.tests.test_utils import get_test_devices

# wp.config.verbose = True
# wp.config.verbose_warnings = True
# wp.config.mode = "debug"

# compute roots for all groups
@wp.kernel
def compute_group_roots(bvh: wp.uint64, group_roots: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    root = wp.bvh_get_group_root(bvh, wp.int32(tid))
    group_roots[tid] = root

@wp.kernel
def test_ray_query(
    bvh: wp.uint64,
    query_start: wp.vec3,
    query_dir: wp.vec3,
    group_roots: wp.array(dtype=wp.int32),
    bounds_intersected: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    root = group_roots[tid]
    query = wp.bvh_query_ray(bvh, query_start, query_dir, root)
    bounds_nr = int(0)
    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1

@wp.kernel
def test_aabb_query(
    bvh: wp.uint64,
    query_lower: wp.vec3,
    query_upper: wp.vec3,
    group_roots: wp.array(dtype=wp.int32),
    bounds_intersected: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    root = group_roots[tid]
    query = wp.bvh_query_aabb(bvh, query_lower, query_upper, root)
    while wp.bvh_query_next(query, tid):
        bounds_intersected[tid] = 1

def test_bvh_with_known_boxes():
    """Test BVH with known bounding boxes that should produce predictable results"""
    devices = get_test_devices()
    device = devices[1]  # Use CPU for easier debugging
    print(f"Testing on device: {device}")

    # Create a simple test case with 2 groups, each with 2 bounding boxes
    # Group 0: boxes at (0,0,0)-(1,1,1) and (2,2,2)-(3,3,3)
    # Group 1: boxes at (4,4,4)-(5,5,5) and (6,6,6)-(7,7,7)
    
    num_groups = 2
    bounds_per_group = 2
    num_bounds = num_groups * bounds_per_group
    
    # Create known bounding boxes
    lowers = np.array([
        # Group 0
        [0.0, 0.0, 0.0],  # Box 0
        [2.0, 2.0, 2.0],  # Box 1
        # Group 1  
        [4.0, 4.0, 4.0],  # Box 2
        [6.0, 6.0, 6.0],  # Box 3
    ], dtype=np.float32)
    
    uppers = np.array([
        # Group 0
        [1.0, 1.0, 1.0],  # Box 0
        [3.0, 3.0, 3.0],  # Box 1
        # Group 1
        [5.0, 5.0, 5.0],  # Box 2
        [7.0, 7.0, 7.0],  # Box 3
    ], dtype=np.float32)
    
    groups = np.array([0, 0, 1, 1], dtype=np.int32)
    
    print("Bounding boxes:")
    for i in range(num_bounds):
        print(f"  Box {i} (Group {groups[i]}): ({lowers[i]}) to ({uppers[i]})")
    
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)

    print('\nBuilding BVH...')
    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)
    print('BVH built successfully')

    # Test group root computation
    print('\nTesting group roots...')
    group_roots = wp.zeros(shape=(num_groups), dtype=wp.int32, device=device)
    wp.launch(compute_group_roots,
        dim=num_groups,
        inputs=[bvh.id, group_roots],
        device=device)
    
    print(f"Group roots: {group_roots.numpy()}")
    
    # Expected: Each group should have its own root, not -1
    for i in range(num_groups):
        root = group_roots.numpy()[i]
        print(f"  Group {i} root: {root}")
        if root == -1:
            print(f"    WARNING: Group {i} has root -1!")
        else:
            print(f"    OK: Group {i} has valid root {root}")

    # Test ray queries
    print('\nTesting ray queries...')
    
    # Ray 1: Should hit Group 0 (boxes 0 and 1)
    query_start = wp.vec3(-1.0, -1.0, -1.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))  # Diagonal ray
    bounds_intersected = wp.zeros(shape=(num_groups), dtype=wp.int32, device=device)
    
    wp.launch(test_ray_query,
        dim=num_groups,
        inputs=[bvh.id, query_start, query_dir, group_roots, bounds_intersected],
        device=device)
    
    print(f"Ray 1 (diagonal from origin) results: {bounds_intersected.numpy()}")
    expected_ray1 = [1, 0]  # Should hit Group 0, not Group 1
    for i in range(num_groups):
        result = bounds_intersected.numpy()[i]
        expected = expected_ray1[i]
        print(f"  Group {i}: got {result}, expected {expected}")
        if result != expected:
            print(f"    WARNING: Group {i} ray query result mismatch!")
    
    # Ray 2: Should hit Group 1 (boxes 2 and 3)
    query_start = wp.vec3(3.0, 3.0, 3.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))  # Diagonal ray
    bounds_intersected = wp.zeros(shape=(num_groups), dtype=wp.int32, device=device)
    
    wp.launch(test_ray_query,
        dim=num_groups,
        inputs=[bvh.id, query_start, query_dir, group_roots, bounds_intersected],
        device=device)
    
    print(f"Ray 2 (diagonal from (3,3,3)) results: {bounds_intersected.numpy()}")
    expected_ray2 = [0, 1]  # Should hit Group 1, not Group 0
    for i in range(num_groups):
        result = bounds_intersected.numpy()[i]
        expected = expected_ray2[i]
        print(f"  Group {i}: got {result}, expected {expected}")
        if result != expected:
            print(f"    WARNING: Group {i} ray query result mismatch!")

    # Test AABB queries
    print('\nTesting AABB queries...')
    
    # AABB 1: Should intersect Group 0
    query_lower = wp.vec3(0.5, 0.5, 0.5)
    query_upper = wp.vec3(2.5, 2.5, 2.5)
    bounds_intersected = wp.zeros(shape=(num_groups), dtype=wp.int32, device=device)
    
    wp.launch(test_aabb_query,
        dim=num_groups,
        inputs=[bvh.id, query_lower, query_upper, group_roots, bounds_intersected],
        device=device)
    
    print(f"AABB 1 (overlaps Group 0) results: {bounds_intersected.numpy()}")
    expected_aabb1 = [1, 0]  # Should hit Group 0, not Group 1
    for i in range(num_groups):
        result = bounds_intersected.numpy()[i]
        expected = expected_aabb1[i]
        print(f"  Group {i}: got {result}, expected {expected}")
        if result != expected:
            print(f"    WARNING: Group {i} AABB query result mismatch!")
    
    # AABB 2: Should intersect Group 1
    query_lower = wp.vec3(4.5, 4.5, 4.5)
    query_upper = wp.vec3(6.5, 6.5, 6.5)
    bounds_intersected = wp.zeros(shape=(num_groups), dtype=wp.int32, device=device)
    
    wp.launch(test_aabb_query,
        dim=num_groups,
        inputs=[bvh.id, query_lower, query_upper, group_roots, bounds_intersected],
        device=device)
    
    print(f"AABB 2 (overlaps Group 1) results: {bounds_intersected.numpy()}")
    expected_aabb2 = [0, 1]  # Should hit Group 1, not Group 0
    for i in range(num_groups):
        result = bounds_intersected.numpy()[i]
        expected = expected_aabb2[i]
        print(f"  Group {i}: got {result}, expected {expected}")
        if result != expected:
            print(f"    WARNING: Group {i} AABB query result mismatch!")

    print('\nTest completed!')

def test_bvh_edge_cases():
    """Test edge cases that might cause issues"""
    devices = get_test_devices()
    device = devices[1]
    print(f"\nTesting edge cases on device: {device}")

    # Test case 1: Single group with multiple boxes
    print("\nTest 1: Single group with multiple boxes")
    lowers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    uppers = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    groups = np.array([0, 0, 0], dtype=np.int32)
    
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)
    
    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)
    
    group_roots = wp.zeros(shape=(1), dtype=wp.int32, device=device)
    wp.launch(compute_group_roots, dim=1, inputs=[bvh.id, group_roots], device=device)
    print(f"Single group root: {group_roots.numpy()[0]}")

    # Test case 2: Multiple groups with single boxes
    print("\nTest 2: Multiple groups with single boxes")
    lowers = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32)
    uppers = np.array([[1.0, 1.0, 1.0], [11.0, 11.0, 11.0]], dtype=np.float32)
    groups = np.array([0, 1], dtype=np.int32)
    
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = wp.array(groups, dtype=wp.int32, device=device)
    
    bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups)
    
    group_roots = wp.zeros(shape=(2), dtype=wp.int32, device=device)
    wp.launch(compute_group_roots, dim=2, inputs=[bvh.id, group_roots], device=device)
    print(f"Two groups roots: {group_roots.numpy()}")

if __name__ == "__main__":
    test_bvh_with_known_boxes()
    test_bvh_edge_cases() 