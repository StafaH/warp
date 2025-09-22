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

#ifdef __CUDA_ARCH__
#define BVH_SHARED_STACK 1
#else
#define BVH_SHARED_STACK 0
#endif

#define BVH_LEAF_SIZE (1)
#define SAH_NUM_BUCKETS (16)
#define USE_LOAD4
#define BVH_QUERY_STACK_SIZE (32)

#define BVH_CONSTRUCTOR_SAH (0)
#define BVH_CONSTRUCTOR_MEDIAN (1)
#define BVH_CONSTRUCTOR_LBVH (2)

#ifndef WP_BVH_BLOCK_DIM
#define WP_BVH_BLOCK_DIM 256
#endif

#define QBVH_WIDTH (4)
#define QBVH_QUANT_BITS (8)
#define QBVH_QUANT_MAX ((1u<<QBVH_QUANT_BITS)-1u)
#define QBVH_INVALID (0xFFFFFFFFu)
#define QBVH_LEAF_BIT (0x80000000u)
#define QBVH_INDEX_MASK (0x7FFFFFFFu)
#define WP_ENABLE_QBVH 1

namespace wp
{

CUDA_CALLABLE inline uint32_t qbvh_leaf_node(uint32_t lbvh_index) {
    return (lbvh_index | QBVH_LEAF_BIT);
}
CUDA_CALLABLE inline bool qbvh_is_leaf(uint32_t tagged) {
    return (tagged & QBVH_LEAF_BIT) != 0;
}
CUDA_CALLABLE inline uint32_t qbvh_leaf_lbvh_index(uint32_t tagged) {
    return (tagged & QBVH_INDEX_MASK);
}

CUDA_CALLABLE inline unsigned char qfloor(float t) {
    int v = (int)floorf(t);
    if (v < 0) v = 0;
    if (v > (int)QBVH_QUANT_MAX) v = QBVH_QUANT_MAX;
    return (unsigned char)v;
}
CUDA_CALLABLE inline unsigned char qceil(float t) {
    int v = (int)ceilf(t);
    if (v < 0) v = 0;
    if (v > (int)QBVH_QUANT_MAX) v = QBVH_QUANT_MAX;
    return (unsigned char)v;
}

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


struct QBVHNode {
    // Typically used to store the indices of the children nodes
    // for leaf nodes, [0] and [1] are the primitive start and end indices
    // We place the children indices at beginning to abuse a float2 cast load for
    // leaf nodes
    uint32_t children_idx[QBVH_WIDTH];
	wp::vec3f min_point;
	wp::vec3f inv_scale;

	uint8_t qminx[QBVH_WIDTH];
	uint8_t qminy[QBVH_WIDTH];
	uint8_t qminz[QBVH_WIDTH];
	uint8_t qmaxx[QBVH_WIDTH];
	uint8_t qmaxy[QBVH_WIDTH];
	uint8_t qmaxz[QBVH_WIDTH];
    uint8_t num_children;
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

    QBVHNode* qbvh_nodes;

    // cuda context
    void* context;
};


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
    float4 f4 = __ldg((const float4*)(nodes)+index);
    return  (const wp::BVHPackedNodeHalf&)f4;
    //return  (const wp::BVHPackedNodeHalf&)(*((const float4*)(nodes)+index));
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

// represents a strided stack in shared memory
// so each level of the stack is stored contiguously
// across the block
struct bvh_stack_t
{
    inline int operator[](int depth) const { return ptr[depth*WP_BVH_BLOCK_DIM]; }
    inline int& operator[](int depth) { return ptr[depth*WP_BVH_BLOCK_DIM]; }

    int* ptr;

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
          primitive_counter(-1),
          leaf_end(-1)
    {}

    // Required for adjoint computations.
    CUDA_CALLABLE inline bvh_query_t& operator+=(const bvh_query_t& other)
    {
        return *this;
    }

    BVH bvh;

    // BVH traversal stack:
#if BVH_SHARED_STACK
    bvh_stack_t stack;
#else
    int stack[BVH_QUERY_STACK_SIZE];
#endif
    int count;

    int primitive_counter;   // >= 0 if currently in a packed leaf node
    int leaf_end;            // exclusive end in leaf-space (valid when primitive_counter>=0)
    
    // inputs
    wp::vec3 input_lower;	// start for ray
    wp::vec3 input_upper;	// dir for ray

    int bounds_nr;
    bool is_ray;
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

CUDA_CALLABLE inline int first_set_bit(uint32_t mask)
{
#ifdef __CUDA_ARCH__
    return mask ? (__ffs(mask) - 1) : -1;
#else
    if (!mask) return -1;
    int idx = 0;
    while ((mask & 1u) == 0u) { mask >>= 1; ++idx; }
    return idx;
#endif
}


CUDA_CALLABLE inline bvh_query_t bvh_query(
	uint64_t id, bool is_ray, const vec3& lower, const vec3& upper, int root)
{
    // This routine traverses the BVH tree until it finds
    // the first overlapping bound. 

    // initialize empty
    bvh_query_t query;

#if BVH_SHARED_STACK
    __shared__ int stack[BVH_QUERY_STACK_SIZE*WP_BVH_BLOCK_DIM];
    query.stack.ptr = &stack[threadIdx.x];
#endif

    query.bounds_nr = -1;

    BVH bvh = bvh_get(id);

    query.bvh = bvh;
    query.is_ray = is_ray;

	// optimization: make the latest	
	query.stack[0] = root == -1 ? *bvh.root : root;
	query.count = 1;
	query.input_lower = lower;
	query.input_upper = upper;

#if WP_ENABLE_QBVH
    while (query.count)
    {
        const int node_index = query.stack[--query.count];
        QBVHNode node = bvh.qbvh_nodes[node_index];

        float dx = query.input_upper[0] / node.inv_scale[0];
        float dy = query.input_upper[1] / node.inv_scale[1];
        float dz = query.input_upper[2] / node.inv_scale[2];
        float ox = (node.min_point[0] - query.input_lower[0]) * query.input_upper[0];
        float oy = (node.min_point[1] - query.input_lower[1]) * query.input_upper[1];
        float oz = (node.min_point[2] - query.input_lower[2]) * query.input_upper[2];

        uint32_t m = 0u; 
        #pragma unroll 
        for (int i = 0; i < (int)node.num_children; ++i)
        {
            const float tnx = (float)node.qminx[i] * dx + ox;
            const float tny = (float)node.qminy[i] * dy + oy;
            const float tnz = (float)node.qminz[i] * dz + oz;
            const float tfx = (float)node.qmaxx[i] * dx + ox;
            const float tfy = (float)node.qmaxy[i] * dy + oy;
            const float tfz = (float)node.qmaxz[i] * dz + oz;
            const float tnear = fmaxf(fminf(tnx, tfx), fmaxf(fminf(tny, tfy), fmaxf(fminf(tnz, tfz), 0.0f)));
            const float tfar = fminf(fmaxf(tfx, tnx), fminf(fmaxf(tfy, tny), fminf(fmaxf(tfz, tnz), 1e30f)));
            if (tnear <= tfar){
                m |= (1u << i); 
            }
        }

        if (m == 0) {
            continue;
        }

        while (m) {
            const int c = first_set_bit(m); m &= (m - 1u);
            const uint32_t child = node.children_idx[c];
            if (qbvh_is_leaf(child))
            {
                // prime iterator: indicate we're about to read a leaf
                query.primitive_counter = 0;
                query.stack[query.count++] = (int)child;
                return query;
            }
            else
            {
                // defer processing internal children to iter
                query.stack[query.count++] = (int)child;
            }
        }
    }
    return query;
#else
    // Navigate through the LBVH, find the first overlapping leaf node.
    while (query.count)
    {
        const int node_index = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
        BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

        if (!bvh_query_intersection_test(query, reinterpret_cast<vec3&>(node_lower), reinterpret_cast<vec3&>(node_upper)))
        {
            continue;
        }

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;
        // Make bounds from this AABB
        if (node_lower.b)
        {
            // Reached a leaf node, point to its first primitive
            // Back up one level and return 
            query.primitive_counter = 0;
            query.stack[query.count++] = node_index;
            return query;
        }
        else
        {
            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
    return query;
#endif // WP_ENABLE_QBVH
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

#if WP_ENABLE_QBVH
    while (query.count)
    {
        const uint32_t tagged = (uint32_t)query.stack[--query.count];
        if (qbvh_is_leaf(tagged))
        {
            const int c = (int)qbvh_leaf_lbvh_index(tagged);
            int start = bvh.qbvh_nodes[c].children_idx[0];
            int end = bvh.qbvh_nodes[c].children_idx[1];
            int primitive_index = bvh.primitive_indices[start + (query.primitive_counter++)];
            // if already visited the last primitive in the leaf node
            // move to the next node and reset the primitive counter to 0
            if (start + query.primitive_counter == end)
            {
                query.primitive_counter = 0;
            }
            // otherwise we need to keep this leaf node in stack for a future visit
            else
            {
                query.stack[query.count++] = tagged;
            }
            
            if (bvh_query_intersection_test(
                query, bvh.item_lowers[primitive_index], bvh.item_uppers[primitive_index]))
            {
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }
        }
        else
        {
            const QBVHNode node   = bvh.qbvh_nodes[tagged];
    
            float dx = query.input_upper[0] / node.inv_scale[0];
            float dy = query.input_upper[1] / node.inv_scale[1];
            float dz = query.input_upper[2] / node.inv_scale[2];
            float ox = (node.min_point[0] - query.input_lower[0]) * query.input_upper[0];
            float oy = (node.min_point[1] - query.input_lower[1]) * query.input_upper[1];
            float oz = (node.min_point[2] - query.input_lower[2]) * query.input_upper[2];
            
            uint32_t m = 0u; 
            #pragma unroll 
            for (int i = 0; i < (int)node.num_children; ++i)
            {
                const float tnx = (float)node.qminx[i] * dx + ox;
                const float tny = (float)node.qminy[i] * dy + oy;
                const float tnz = (float)node.qminz[i] * dz + oz;
                const float tfx = (float)node.qmaxx[i] * dx + ox;
                const float tfy = (float)node.qmaxy[i] * dy + oy;
                const float tfz = (float)node.qmaxz[i] * dz + oz;
                const float tnear = fmaxf(fminf(tnx, tfx), fmaxf(fminf(tny, tfy), fmaxf(fminf(tnz, tfz), 0.0f)));
                const float tfar = fminf(fmaxf(tfx, tnx), fminf(fmaxf(tfy, tny), fminf(fmaxf(tfz, tnz), 1e30f)));
                if (tnear <= tfar) m |= (1u << i);
            }
            while (m) {
                const int c = first_set_bit(m);
                m &= (m - 1u);
                uint32_t child = node.children_idx[c];
                query.stack[query.count++] = (int)child;
            }
        }
    }
    return false;
#else // WP_ENABLE_QBVH

    // Navigate through the LBVH, find the first overlapping leaf node.
    while (query.count)
    {
        const int node_index = query.stack[--query.count];

        BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
        BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

        if (!bvh_query_intersection_test(query, reinterpret_cast<vec3&>(node_lower), reinterpret_cast<vec3&>(node_upper)))
        {
            continue;
        }

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        if (node_lower.b)
        {
            // found leaf
            const int left_index  = node_lower.i;
            const int right_index = node_upper.i;
        
            if (right_index == left_index) {
                // original leaf: single primitive = primitive_indices[leaf_index]
                const int prim = bvh.primitive_indices[node_index];
                index = prim;
                query.bounds_nr = prim;
                return true;
            } else {
                // packed leaf range
                query.primitive_counter = left_index;
                query.leaf_end = right_index;
        
                const int prim = bvh.primitive_indices[query.primitive_counter++];
                index = prim;
                query.bounds_nr = prim;
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
#endif // WP_ENABLE_QBVH
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


void bvh_create_host(vec3* lowers, vec3* uppers, int num_items,  int constructor_type, int* groups, BVH& bvh);
void bvh_destroy_host(wp::BVH& bvh);
void bvh_refit_host(wp::BVH& bvh);

#if WP_ENABLE_CUDA

void bvh_create_device(void* context, vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, BVH& bvh_device_on_host);
void bvh_destroy_device(BVH& bvh);
void bvh_refit_device(BVH& bvh);

#endif // WP_ENABLE_CUDA

} // namespace wp
