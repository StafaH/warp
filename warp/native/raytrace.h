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

#if defined(__CUDA_ARCH__)
#define WP_ALIGNED(N) __align__(N)
#else
#define WP_ALIGNED(N) alignas(N)
#endif

namespace wp
{

struct BatchRenderer
{
    int img_width;
    int img_height;
    int nworld;
    int ncam;
    float fov_rad;

    // geometry
    int ngeom;
    struct Geom* geoms;

    // cuda context
    void* context;

    inline CUDA_CALLABLE BatchRenderer(int id = 0)
    {
        img_width = 0;
        img_height = 0;
        nworld = 0;
        ncam = 0;
        fov_rad = 0;
        ngeom = 0;
        geoms = nullptr;
        context = nullptr;
    }

    inline CUDA_CALLABLE BatchRenderer(
        int img_width,
        int img_height,
        int nworld,
        int ncam,
        float fov_rad,
        int ngeom = 0,
        struct Geom* geoms = nullptr,
        void* context = nullptr
    ) : img_width(img_width), img_height(img_height), nworld(nworld), ncam(ncam), fov_rad(fov_rad), ngeom(ngeom), geoms(geoms), context(context)
    {
    }
};

CUDA_CALLABLE inline BatchRenderer batch_renderer_get(uint64_t id)
{
    return *(BatchRenderer*)(id);
}


CUDA_CALLABLE inline void compute_camera_ray(
    int width,
    int height,
    float fov_rad,
    int px,
    int py,
    const wp::mat33& cam_xmat,
    wp::vec3& out_ray_dir)
{
    float aspect = float(width) / float(height);
    float u = (float(px) + 0.5f) / float(width) - 0.5f;
    float v = (float(py) + 0.5f) / float(height) - 0.5f;
    float h = tanf(fov_rad * 0.5f);
    float rx = u * 2.0f * h;
    float ry = -v * 2.0f * h / aspect;
    float rz = -1.0f;

    wp::vec3 dir_local_cam = wp::normalize(wp::vec3(rx, ry, rz));
    wp::vec3 d = wp::mul(cam_xmat, dir_local_cam);
    out_ray_dir = d;
}

bool batch_renderer_get_descriptor(uint64_t id, BatchRenderer& r);
bool batch_renderer_set_descriptor(uint64_t id, const BatchRenderer& r);
void batch_renderer_add_descriptor(uint64_t id, const BatchRenderer& r);
void batch_renderer_rem_descriptor(uint64_t id);

// -----------------------------------------------------------------------------
// Ray intersection helpers and functions (CUDA-friendly)
// -----------------------------------------------------------------------------

// Small epsilon for numerical robustness
static constexpr float WP_RAY_EPS = 1.0e-8f;

// Packed geometry description
struct WP_ALIGNED(16) Geom
{
    int type;
    wp::vec3 size;

    CUDA_CALLABLE inline Geom() : type(0), size(0.0f) {}
    CUDA_CALLABLE inline Geom(int t, const wp::vec3& s) : type(t), size(s) {}
};

// Map a world-space ray to the local frame defined by position and orientation
CUDA_CALLABLE inline void ray_map(
    const wp::vec3& pos,
    const wp::mat33& mat,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    wp::vec3& out_lpnt,
    wp::vec3& out_lvec)
{
    const wp::mat33 matT = wp::transpose(mat);
    out_lpnt = wp::mul(matT, (pnt - pos));
    out_lvec = wp::mul(matT, dir);
}

// Solve a quadratic a*x^2 + 2*b*x + c = 0
// Returns the smallest non-negative root in ret_root if any, and both roots in out_roots
CUDA_CALLABLE inline bool ray_quad(
    float a,
    float b,
    float c,
    float& ret_root,
    wp::vec2& out_roots)
{
    const float det = b*b - a*c;
    if (det < WP_RAY_EPS)
    {
        out_roots = wp::vec2(FLT_MAX, FLT_MAX);
        ret_root = FLT_MAX;
        return false;
    }

    const float sdet = wp::sqrt(det);
    const float inv_a = 1.0f / a;
    const float x0 = (-b - sdet) * inv_a;
    const float x1 = (-b + sdet) * inv_a;
    out_roots = wp::vec2(x0, x1);

    float x = FLT_MAX;
    if (x0 >= 0.0f) x = x0; else if (x1 >= 0.0f) x = x1;
    ret_root = x;
    return x != FLT_MAX;
}

// Plane: local plane is z=0, front face along +Z in local coords.
// size: half-extents in X and Y; <= 0 means infinite in that axis.
CUDA_CALLABLE inline bool ray_plane(
    const wp::vec3& pos,
    const wp::mat33& mat,
    const wp::vec3& size,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    float& out_t,
    wp::vec3& out_normal)
{
    wp::vec3 lp, ld;
    ray_map(pos, mat, pnt, dir, lp, ld);

    if (ld[2] > -WP_RAY_EPS)
        return false;

    const float t = -lp[2] / ld[2];
    if (t < 0.0f)
        return false;

    const float px = lp[0] + t*ld[0];
    const float py = lp[1] + t*ld[1];
    const bool in_x = (size[0] <= 0.0f) || (wp::abs(px) <= size[0]);
    const bool in_y = (size[1] <= 0.0f) || (wp::abs(py) <= size[1]);
    if (!(in_x && in_y))
        return false;

    out_t = t;
    wp::vec3 n_local = wp::vec3(0.0f, 0.0f, 1.0f);
    out_normal = wp::normalize(wp::mul(mat, n_local));
    return true;
}

// Sphere centered at pos with radius
CUDA_CALLABLE inline bool ray_sphere(
    const wp::vec3& pos,
    float radius,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    float& out_t,
    wp::vec3& out_normal)
{
    const wp::vec3 dif = pnt - pos;
    const float a = wp::dot(dir, dir);
    const float b = wp::dot(dir, dif);
    const float c = wp::dot(dif, dif) - radius*radius;
    wp::vec2 roots;
    float t;
    if (!ray_quad(a, b, c, t, roots) || t == FLT_MAX)
        return false;

    out_t = t;
    out_normal = wp::normalize(pnt + t*dir - pos);
    return true;
}

// Capsule aligned with local Z axis; size.x = radius, size.y = half-height
CUDA_CALLABLE inline bool ray_capsule(
    const wp::vec3& pos,
    const wp::mat33& mat,
    const wp::vec3& size,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    float& out_t,
    wp::vec3& out_normal)
{
    const float radius = size[0];
    const float half_h = size[1];

    // quick bounding sphere reject
    float t_sphere;
    wp::vec3 n_sphere;
    if (!ray_sphere(pos, radius + half_h, pnt, dir, t_sphere, n_sphere))
        return false;
    if (t_sphere < 0.0f)
        return false;

    wp::vec3 lp, ld;
    ray_map(pos, mat, pnt, dir, lp, ld);

    float best_t = FLT_MAX;

    // Cylinder side (x^2 + y^2 = r^2) within |z| <= half_h
    {
        const float a = ld[0]*ld[0] + ld[1]*ld[1];
        const float b = ld[0]*lp[0] + ld[1]*lp[1];
        const float c = lp[0]*lp[0] + lp[1]*lp[1] - radius*radius;
        wp::vec2 roots;
        float t;
        if (a > WP_RAY_EPS && ray_quad(a, b, c, t, roots) && t != FLT_MAX)
        {
            const float z = lp[2] + t*ld[2];
            if (wp::abs(z) <= half_h)
                best_t = wp::min(best_t, t);
        }
    }

    // Spherical caps at z = +/- half_h
    auto test_cap = [&](float cap_z, float& best) {
        const wp::vec3 center = wp::vec3(0.0f, 0.0f, cap_z);
        const wp::vec3 dif = lp - center;
        const float a = wp::dot(ld, ld);
        const float b = wp::dot(ld, dif);
        const float c = wp::dot(dif, dif) - radius*radius;
        wp::vec2 roots; float t;
        if (ray_quad(a, b, c, t, roots) && t != FLT_MAX)
        {
            const float z = lp[2] + t*ld[2];
            if ((cap_z > 0.0f && z >= cap_z - WP_RAY_EPS) || (cap_z < 0.0f && z <= cap_z + WP_RAY_EPS))
            {
                best = wp::min(best, t);
            }
        }
    };
    test_cap( half_h, best_t);
    test_cap(-half_h, best_t);

    if (best_t == FLT_MAX)
        return false;

    out_t = best_t;
    const wp::vec3 hit_local = lp + best_t*ld;
    wp::vec3 n_local;
    if (wp::abs(hit_local[2]) < half_h - 1.0e-6f)
    {
        n_local = wp::normalize(wp::vec3(hit_local[0], hit_local[1], 0.0f));
    }
    else
    {
        const float cap_z = (hit_local[2] >= 0.0f) ? half_h : -half_h;
        n_local = wp::normalize(hit_local - wp::vec3(0.0f, 0.0f, cap_z));
    }
    out_normal = wp::normalize(wp::mul(mat, n_local));
    return true;
}

// Axis-aligned box in local frame with half-extents size; oriented by mat, centered at pos.
CUDA_CALLABLE inline bool ray_box(
    const wp::vec3& pos,
    const wp::mat33& mat,
    const wp::vec3& size,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    float& out_t,
    wp::vec3& out_normal)
{
    // quick bounding sphere reject
    if (wp::dot(size, size) > 0.0f)
    {
        float t_bs; wp::vec3 n_bs;
        if (!ray_sphere(pos, wp::sqrt(wp::dot(size, size)), pnt, dir, t_bs, n_bs) || t_bs < 0.0f)
            return false;
    }

    wp::vec3 lp, ld;
    ray_map(pos, mat, pnt, dir, lp, ld);

    float best_t = FLT_MAX;
    int best_axis = -1;
    float best_side = 0.0f;

    const int IFACE[3][2] = {{1,2},{0,2},{0,1}};
    for (int i = 0; i < 3; ++i)
    {
        if (wp::abs(ld[i]) <= WP_RAY_EPS) continue;
        for (int s = -1; s <= 1; s += 2)
        {
            const float plane = float(s) * size[i];
            const float t = (plane - lp[i]) / ld[i];
            if (t < 0.0f) continue;

            const int id0 = IFACE[i][0];
            const int id1 = IFACE[i][1];
            const float p0 = lp[id0] + t*ld[id0];
            const float p1 = lp[id1] + t*ld[id1];
            if (wp::abs(p0) <= size[id0] + WP_RAY_EPS && wp::abs(p1) <= size[id1] + WP_RAY_EPS)
            {
                if (t < best_t)
                {
                    best_t = t;
                    best_axis = i;
                    best_side = float(s);
                }
            }
        }
    }

    if (best_axis < 0)
        return false;

    out_t = best_t;
    wp::vec3 n_local(0.0f);
    n_local[best_axis] = -wp::sign(best_side);
    out_normal = wp::normalize(wp::mul(mat, n_local));
    return true;
}

// Single triangle intersection using Woop ray-tri for robustness
CUDA_CALLABLE inline bool ray_triangle(
    const wp::vec3& v0,
    const wp::vec3& v1,
    const wp::vec3& v2,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    float& out_t,
    wp::vec3& out_normal)
{
    float t, u, v, sign;
    wp::vec3 n;
    if (!intersect_ray_tri_woop(pnt, dir, v0, v1, v2, t, u, v, sign, &n))
        return false;
    if (t < 0.0f)
        return false;
    out_t = t;
    // Orient normal against the ray
    wp::vec3 nn = wp::normalize(n);
    if (wp::dot(nn, dir) > 0.0f) nn = -nn;
    out_normal = nn;
    return true;
}

// Mesh ray intersection: transforms ray into local mesh frame, queries BVH, backface culls in local space
CUDA_CALLABLE inline bool ray_mesh(
    const wp::array_t<wp::uint64>& mesh_bvh_ids,
    int mesh_id,
    const wp::vec3& pos,
    const wp::mat33& mat,
    const wp::vec3& pnt,
    const wp::vec3& dir,
    float max_t,
    float& out_t,
    wp::vec3& out_normal,
    float& out_u,
    float& out_v,
    int& out_face,
    int& out_mesh_id)
{
    // wp::vec3 lp, ld;
    // ray_map(pos, mat, pnt, dir, lp, ld);

    // float t = max_t;
    // float u = 0.0f, v = 0.0f, sign = 0.0f;
    // wp::vec3 n;
    // int f = -1;

    // const bool hit = mesh_query_ray(mesh_bvh_ids[mesh_id], lp, ld, max_t, t, u, v, sign, n, f);
    // if (hit && wp::dot(ld, n) < 0.0f)
    // {
    //     out_t = t;
    //     out_u = u;
    //     out_v = v;
    //     out_face = f;
    //     out_mesh_id = mesh_id;
    //     out_normal = wp::normalize(wp::mul(mat, n));
    //     return true;
    // }

    out_t = FLT_MAX;
    out_u = 0.0f;
    out_v = 0.0f;
    out_face = -1;
    out_mesh_id = -1;
    out_normal = wp::vec3(0.0f);
    return false;
}

} // namespace wp



