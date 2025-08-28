/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "builtin.h"
#include "intersect.h"

// Fallback implementation of __ffs for both CUDA and non-CUDA compilation
CUDA_CALLABLE inline int __ffs(int x) {
    if (x == 0) return 0;
    int result = 1;
    while ((x & 1) == 0) {
        x >>= 1;
        result++;
    }
    return result;
}

#define BVH_LEAF_SIZE (4)
#define SAH_NUM_BUCKETS (16)
#define USE_LOAD4
#define BVH_QUERY_STACK_SIZE (32)
#define WP_BVH_QBVH_WIDTH (8)

#define BVH_CONSTRUCTOR_SAH (0)
#define BVH_CONSTRUCTOR_MEDIAN (1)
#define BVH_CONSTRUCTOR_LBVH (2)

namespace wp
{

struct bounds3
{
	CUDA_CALLABLE inline bounds3() : lower( FLT_MAX)
						           , upper(-FLT_MAX) {}

	CUDA_CALLABLE inline bounds3(const vec3& lower, const vec3& upper) : lower(lower), upper(upper) {}

	CUDA_CALLABLE inline vec3 center() const { return 0.5f*(lower+upper); }
	CUDA_CALLABLE inline vec3 edges() const { return upper-lower; }

	CUDA_CALLABLE inline void expand(float r)
	{
		lower -= vec3(r);
		upper += vec3(r);
	}

	CUDA_CALLABLE inline void expand(const vec3& r)
	{
		lower -= r;
		upper += r;
	}

	CUDA_CALLABLE inline bool empty() const { return lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2]; }

	CUDA_CALLABLE inline bool overlaps(const vec3& p) const
	{
		if (p[0] < lower[0] ||
			p[1] < lower[1] ||
			p[2] < lower[2] ||
			p[0] > upper[0] ||
			p[1] > upper[1] ||
			p[2] > upper[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline bool overlaps(const bounds3& b) const
	{
		if (lower[0] > b.upper[0] ||
			lower[1] > b.upper[1] ||
			lower[2] > b.upper[2] ||
			upper[0] < b.lower[0] ||
			upper[1] < b.lower[1] ||
			upper[2] < b.lower[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline bool overlaps(const vec3& b_lower, const vec3& b_upper) const
	{
		if (lower[0] > b_upper[0] ||
			lower[1] > b_upper[1] ||
			lower[2] > b_upper[2] ||
			upper[0] < b_lower[0] ||
			upper[1] < b_lower[1] ||
			upper[2] < b_lower[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline void add_point(const vec3& p)
	{
		lower = min(lower, p);
		upper = max(upper, p);
	}

	CUDA_CALLABLE inline void add_bounds(const vec3& lower_other, const vec3& upper_other)
	{
		// lower_other will only impact the lower of the new bounds
		// upper_other will only impact the upper of the new bounds
		// this costs only half of the computation of adding lower_other and upper_other separately
		lower = min(lower, lower_other);
		upper = max(upper, upper_other);
	}

	CUDA_CALLABLE inline float area() const
	{
		vec3 e = upper-lower;
		return 2.0f*(e[0]*e[1] + e[0]*e[2] + e[1]*e[2]);
	}

	vec3 lower;
	vec3 upper;
};

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const vec3& b) 
{
	return bounds3(min(a.lower, b), max(a.upper, b));
}

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const bounds3& b) 
{
	return bounds3(min(a.lower, b.lower), max(a.upper, b.upper));
}

CUDA_CALLABLE inline bounds3 bounds_intersection(const bounds3& a, const bounds3& b)
{
	return bounds3(max(a.lower, b.lower), min(a.upper, b.upper));
}

struct BVHPackedNodeHalf
{
	float x;
	float y;
	float z;
	// For non-leaf nodes:
	// - 'lower.i' represents the index of the left child node.
	// - 'upper.i' represents the index of the right child node.
	//
	// For leaf nodes:
	// - 'lower.i' indicates the start index of the primitives in 'primitive_indices'.
	// - 'upper.i' indicates the index just after the last primitive in 'primitive_indices'
	unsigned int i : 31;
	unsigned int b : 1;
};

struct QBVHNode
{
	uint32_t children_idx[WP_BVH_QBVH_WIDTH];   // internal: QBVH index | 0x8000_0000u for leaf sentinel
	uint8_t  num_children;

	vec3f min_point;
	vec3f inv_scale;

	uint8_t qminx[WP_BVH_QBVH_WIDTH];
	uint8_t qminy[WP_BVH_QBVH_WIDTH];
	uint8_t qminz[WP_BVH_QBVH_WIDTH];
	uint8_t qmaxx[WP_BVH_QBVH_WIDTH];
	uint8_t qmaxy[WP_BVH_QBVH_WIDTH];
	uint8_t qmaxz[WP_BVH_QBVH_WIDTH];
};

struct BVH
{		
    BVHPackedNodeHalf* node_lowers;
    BVHPackedNodeHalf* node_uppers;

	// used for fast refits
	int* node_parents;
	int* node_counts;
	uint64_t* keys;
	// reordered primitive indices corresponds to the ordering of leaf nodes
	int* primitive_indices;
	
	int max_depth;
	int max_nodes;
	int num_nodes;
	// since we use packed leaf nodes, the number of them is no longer the number of items, but variable
	int num_leaf_nodes;

	// pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
	// representing the root of the tree, this is not always the first node
	// for bottom-up builders
	int* root;
	// pointer for the root of each group
	int* group_roots;

	// item bounds are not owned by the BVH but by the caller
	vec3* item_lowers;
	vec3* item_uppers;
	int* item_groups;
	int num_items;

	// cuda context
	void* context;
    
	// optional widened/quantized nodes
	QBVHNode* qnodes;
	int qnum_nodes;
	unsigned flags;
};

CUDA_CALLABLE inline bool bvh_has_qbvh(const BVH& b)
{
	return (b.flags & 1u) && b.qnodes && b.qnum_nodes > 0;
}

CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound[0];
    n.y = bound[1];
    n.z = bound[2];
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound[0];
    n->y = bound[1];
    n->z = bound[2];
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf?1:0);
}

#ifdef __CUDA_ARCH__
__device__ inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
{
#ifdef USE_LOAD4
	//return  (const wp::BVHPackedNodeHalf&)(__ldg((const float4*)(nodes)+index));
	return  (const wp::BVHPackedNodeHalf&)(*((const float4*)(nodes)+index));
#else
	return  nodes[index];
#endif // USE_LOAD4

}
#else
inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
{
	return  nodes[index];
}
#endif // __CUDACC__

CUDA_CALLABLE inline bool qbvh_is_leaf(uint32_t c) { return (c & 0x80000000u) != 0; }

CUDA_CALLABLE inline uint32_t qbvh_leaf_node(uint32_t lbvh_node_idx) { return (lbvh_node_idx | 0x80000000u); }

CUDA_CALLABLE inline uint32_t qbvh_child_index(uint32_t c) { return (c & ~0x80000000u); }

CUDA_CALLABLE inline int clz(int x)
{
    int n;
    if (x == 0) return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
    return n;
}

CUDA_CALLABLE inline uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;

    return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*lwp2(dim) bits 
template <int dim>
CUDA_CALLABLE inline uint32_t morton3(float x, float y, float z)
{
    uint32_t ux = clamp(int(x*dim), 0, dim-1);
    uint32_t uy = clamp(int(y*dim), 0, dim-1);
    uint32_t uz = clamp(int(z*dim), 0, dim-1);

    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

// making the class accessible from python

CUDA_CALLABLE inline BVH bvh_get(uint64_t id)
{
    return *(BVH*)(id);
}

CUDA_CALLABLE inline int bvh_get_num_bounds(uint64_t id)
{
	BVH bvh = bvh_get(id);
	return bvh.num_items;
}

CUDA_CALLABLE inline int lower_bound_group(const uint64_t* keys, int n, unsigned int group)
{
	uint64_t prefix = uint64_t(group) << 32;
	int lo = 0;
	int hi = n;
	
	while (lo < hi)
	{
    	int mid = (lo + hi) >> 1;
		if (keys[mid] < prefix)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}
	
	if (lo == n || (keys[lo] >> 32) != group) return -1;
	
	return lo;
}


CUDA_CALLABLE inline int lca(int node_a, int node_b, const int* parent)
{
	int da = 0, db = 0;
    for (int t = node_a; t != -1; t = parent[t]) ++da;
    for (int t = node_b; t != -1; t = parent[t]) ++db;

	if (da > db) {
        int diff = da - db;
        while (diff-- && node_a != -1) node_a = parent[node_a];
    } else if (db > da) {
        int diff = db - da;
        while (diff-- && node_b != -1) node_b = parent[node_b];
    }

	while (node_a != node_b) {
        if (node_a == -1 || node_b == -1) return -1;
        node_a = parent[node_a];
        node_b = parent[node_b];
    }
    return node_a;  // either the LCA or -1
}


CUDA_CALLABLE inline int bvh_get_group_root(uint64_t id, int group_id)
{
	BVH bvh = bvh_get(id);
	// locate first leaf of the current group
	int first = lower_bound_group(bvh.keys, bvh.num_items, group_id);
	if (first < 0) return -1;
	
	// find first leaf of next group to find the last leaf of the current group
    int next = lower_bound_group(bvh.keys, bvh.num_items, group_id + 1);
	int last = (next < 0 ? bvh.num_items : next) - 1;

	// climb both until we meet
	return lca(first, last, bvh.node_parents);
}

struct NodeGroup {
	uint32_t qnode;     // QBVH node index
	uint32_t mask;      // present bits for children (up to 8)
};

// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct bvh_query_t
{
    CUDA_CALLABLE bvh_query_t()
        : bvh(),
          stack(),
          count(0),
          is_ray(false),
          input_lower(),
          input_upper(),
          bounds_nr(0),
		  primitive_counter(-1)
    {}

    // Required for adjoint computations.
    CUDA_CALLABLE inline bvh_query_t& operator+=(const bvh_query_t& other)
    {
        return *this;
    }

    BVH bvh;

	// BVH traversal stack:
	int stack[BVH_QUERY_STACK_SIZE];
	int count;

	// >= 0 if currently in a packed leaf node
	int primitive_counter;
	
    // inputs
    wp::vec3 input_lower;	// start for ray
    wp::vec3 input_upper;	// dir for ray

	int bounds_nr;
	bool is_ray;

	//qbvh
	NodeGroup qstack[BVH_QUERY_STACK_SIZE];
	int qcount = 0;
};

CUDA_CALLABLE inline bool bvh_query_intersection_test(const bvh_query_t& query, const vec3& node_lower, const vec3& node_upper)
{
	if (query.is_ray)
	{
		float t = 0.0f;
		return intersect_ray_aabb(query.input_lower, query.input_upper, node_lower, node_upper, t);
	}
	else
	{
		return intersect_aabb_aabb(query.input_lower, query.input_upper, node_lower, node_upper);
	}
}


CUDA_CALLABLE inline void qbvh_dir_origin_quant(const QBVHNode& n,
																					 const vec3& ray_o,
																					 const vec3& inv_ray_d,
																					 vec3& dirq, vec3& orgq)
{
	// dirq = inv_ray_d / scale ; orgq = (minPoint - ray_o) * inv_ray_d
	dirq[0] = inv_ray_d[0] / n.inv_scale[0];  dirq[1] = inv_ray_d[1] / n.inv_scale[1];  dirq[2] = inv_ray_d[2] / n.inv_scale[2];
	orgq[0] = (n.min_point[0] - ray_o[0]) * inv_ray_d[0];
	orgq[1] = (n.min_point[1] - ray_o[1]) * inv_ray_d[1];
	orgq[2] = (n.min_point[2] - ray_o[2]) * inv_ray_d[2];
}

CUDA_CALLABLE inline uint32_t qbvh_present_mask(const QBVHNode& n,
																					 const vec3& dirq,
																					 const vec3& orgq,
																					 float tmax)
{
	uint32_t m = 0;
#pragma unroll
	for (int i=0;i<n.num_children;i++) {
			float tnx = n.qminx[i]*dirq[0] + orgq[0];
			float tny = n.qminy[i]*dirq[1] + orgq[1];
			float tnz = n.qminz[i]*dirq[2] + orgq[2];
			float tfx = n.qmaxx[i]*dirq[0] + orgq[0];
			float tfy = n.qmaxy[i]*dirq[1] + orgq[1];
			float tfz = n.qmaxz[i]*dirq[2] + orgq[2];

			float tnear = fmaxf(fminf(tnx,tfx), fmaxf(fminf(tny,tfy), fmaxf(fminf(tnz,tfz), 0.f)));
			float tfar  = fminf(fmaxf(tfx,tnx), fminf(fmaxf(tfy,tny), fminf(fmaxf(tfz,tnz), tmax)));
			if (tnear <= tfar) m |= (1u<<i);
	}
	return m;
}

CUDA_CALLABLE inline bvh_query_t bvh_query(
	uint64_t id, bool is_ray, const vec3& lower, const vec3& upper, int root)
{
	// This routine traverses the BVH tree until it finds
	// the first overlapping bound. 

	// initialize empty
	bvh_query_t query;

	query.bounds_nr = -1;

	BVH bvh = bvh_get(id);

	query.bvh = bvh;
	query.is_ray = is_ray;

	// store inputs
	query.input_lower = lower;
	query.input_upper = upper;
	query.primitive_counter = 0;

	// Start traversal: use QBVH only for rays and when available; otherwise fall back to LBVH
	if (is_ray && bvh_has_qbvh(bvh))
	{
		// QBVH root is 0 from widen
		query.qstack[0] = NodeGroup{ (uint32_t)0, 0u };
		query.qcount = 1;
		query.count = 0; // LBVH stack empty; QBVH will enqueue LBVH leaves
	}
	else
	{
		query.qcount = 0;
		query.stack[0] = root == -1 ? *bvh.root : root;
		query.count = 1;
	}
	return query;

	// // optimization: make the latest	
	// query.stack[0] = root == -1 ? *bvh.root : root;
	// query.count = 1;
	// query.input_lower = lower;
	// query.input_upper = upper;

	// // Navigate through the bvh, find the first overlapping leaf node.
	// while (query.count)
	// {
	// 	const int node_index = query.stack[--query.count];
	// 	BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
	// 	BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

	// 	if (!bvh_query_intersection_test(query, (vec3&)node_lower, (vec3&)node_upper))
	// 	{
	// 		continue;
	// 	}

	// 	const int left_index = node_lower.i;
	// 	const int right_index = node_upper.i;
	// 	// Make bounds from this AABB
	// 	if (node_lower.b)
	// 	{
	// 		// Reached a leaf node, point to its first primitive
	// 		// Back up one level and return 
	// 		query.primitive_counter = 0;
	// 		query.stack[query.count++] = node_index;
	// 		return query;
	// 	}
	// 	else
	// 	{
	// 		query.stack[query.count++] = left_index;
	// 		query.stack[query.count++] = right_index;
	// 	}
	// }

	// return query;
}

CUDA_CALLABLE inline bvh_query_t bvh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper, int root)
{
	return bvh_query(id, false, lower, upper, root);
}


CUDA_CALLABLE inline bvh_query_t bvh_query_ray(
    uint64_t id, const vec3& start, const vec3& dir, int root)
{
	return bvh_query(id, true, start, 1.0f / dir, root);
}

//Stub
CUDA_CALLABLE inline void adj_bvh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper,
											   int root, uint64_t, vec3&, vec3&, int&, bvh_query_t&)
{
}


CUDA_CALLABLE inline void adj_bvh_query_ray(uint64_t id, const vec3& start, const vec3& dir,
											   int root, uint64_t, vec3&, vec3&, int&, bvh_query_t&)
{
}

CUDA_CALLABLE inline void adj_bvh_get_group_root(uint64_t id, int group_id, uint64_t&, int&, int&)
{
}


CUDA_CALLABLE inline bool bvh_query_next(bvh_query_t& query, int& index)
{
	BVH bvh = query.bvh;

	// Ray: traverse QBVH first (if available); on QBVH leaf, scan LBVH leaf’s [start,end)
	if (query.is_ray && bvh_has_qbvh(bvh))
	{
		const vec3 ray_o = query.input_lower;
		const vec3 inv_d = query.input_upper;

		while (query.qcount) {
				NodeGroup g = query.qstack[--query.qcount];
				const QBVHNode& n = bvh.qnodes[g.qnode];

				vec3 dirq, orgq;
				qbvh_dir_origin_quant(n, ray_o, inv_d, dirq, orgq);
				// uint32_t present = qbvh_present_mask(n, dirq, orgq, /*tmax=*/1e30f);
				uint32_t present = g.mask ? g.mask : qbvh_present_mask(n, dirq, orgq, /*tmax=*/1e30f);
				
				while (present) {
						int c = __ffs(present) - 1; // find first set bit
						present &= (present - 1); // pop low bit

						uint32_t child = n.children_idx[c];
						if (qbvh_is_leaf(child)) {
								// LBVH leaf: push it on the LBVH stack to reuse your leaf scan logic
								int lbvh_node = (int)qbvh_child_index(child);
								if (query.count < BVH_QUERY_STACK_SIZE) {
									query.primitive_counter = 0; // reset for new leaf
									query.stack[query.count++] = lbvh_node;
								}
								// fall through to LBVH loop below for primitive scan
						} else {
							if (present && query.qcount < BVH_QUERY_STACK_SIZE) {
								query.qstack[query.qcount++] = NodeGroup{ g.qnode, present };
							}
							if (query.qcount < BVH_QUERY_STACK_SIZE) {
								query.qstack[query.qcount++] = NodeGroup{ child, 0u };
							}
							present = 0; // ensure we process child now
						}
				}

				// After consuming all present children of this QBVH node, continue while(qcount)
				// The LBVH scan happens in the shared leaf loop below.
				// break to LBVH only if LBVH stack is non-empty:
				if (query.count) break;
		}
	}
	// If QBVH exhausted, query.count might still have LBVH leafs enqueued

	// Navigate through the bvh, find the first overlapping leaf node.
	while (query.count)
	{
		const int node_index = query.stack[--query.count];

		BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
		BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

		if (!bvh_query_intersection_test(query, (vec3&)node_lower, (vec3&)node_upper))
		{
			continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

		if (node_lower.b)
		{
			// found leaf, loop through its content primitives
			const int start = left_index;
			const int end = right_index;

			// int primitive_index = bvh.primitive_indices[start + (query.primitive_counter++)];
			if (query.primitive_counter < 0) query.primitive_counter = 0;
			const int primitive_index = bvh.primitive_indices[start + (query.primitive_counter++)];
			// if already visited the last primitive in the leaf node
			// move to the next node and reset the primitive counter to 0
			if (start + query.primitive_counter == end)
			{
				query.primitive_counter = 0;
			}
			// otherwise we need to keep this leaf node in stack for a future visit
			else
			{
				query.stack[query.count++] = node_index;
			}
			if (bvh_query_intersection_test(query, bvh.item_lowers[primitive_index], bvh.item_uppers[primitive_index]))
			{
				index = primitive_index;
				query.bounds_nr = primitive_index;

				return true;
			}
		}
		else
		{
			// if it's not a leaf node we treat it as if we have visited the last primitive
			query.primitive_counter = 0;
			query.stack[query.count++] = left_index;
			query.stack[query.count++] = right_index;
		}
	}
	return false;
}


CUDA_CALLABLE inline int iter_next(bvh_query_t& query)
{
    return query.bounds_nr;
}

CUDA_CALLABLE inline bool iter_cmp(bvh_query_t& query)
{
    bool finished = bvh_query_next(query, query.bounds_nr);
    return finished;
}

CUDA_CALLABLE inline bvh_query_t iter_reverse(const bvh_query_t& query)
{
    // can't reverse BVH queries, users should not rely on traversal ordering
    return query;
}

CUDA_CALLABLE inline void adj_iter_reverse(const bvh_query_t& query, bvh_query_t& adj_query, bvh_query_t& adj_ret)
{
}


// stub
CUDA_CALLABLE inline void adj_bvh_query_next(bvh_query_t& query, int& index, bvh_query_t&, int&, bool&) 
{

}

CUDA_CALLABLE bool bvh_get_descriptor(uint64_t id, BVH& bvh);
CUDA_CALLABLE void bvh_add_descriptor(uint64_t id, const BVH& bvh);
CUDA_CALLABLE void bvh_rem_descriptor(uint64_t id);

#if !__CUDA_ARCH__

void bvh_create_host(vec3* lowers, vec3* uppers, int num_items,  int constructor_type, int* groups, BVH& bvh);
void bvh_destroy_host(wp::BVH& bvh);
void bvh_refit_host(wp::BVH& bvh);

void bvh_destroy_device(wp::BVH& bvh);
void bvh_refit_device(uint64_t id);

#endif

} // namespace wp
