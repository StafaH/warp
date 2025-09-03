/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "raytrace.h"
#include "warp.h"
#include "cuda_util.h"

#include <map>

using namespace wp;

namespace 
{
    // host-side copy of batch renderer descriptors, maps GPU batch renderer address (id) to a CPU desc
    std::map<uint64_t, BatchRenderer> g_batch_renderer_descriptors;

} // anonymous namespace


namespace wp
{

bool batch_renderer_get_descriptor(uint64_t id, BatchRenderer& r)
{
    const auto& iter = g_batch_renderer_descriptors.find(id);
    if (iter == g_batch_renderer_descriptors.end())
        return false;
    else
        r = iter->second;
        return true;
}

bool batch_renderer_set_descriptor(uint64_t id, const BatchRenderer& r)
{
    const auto& iter = g_batch_renderer_descriptors.find(id);
    if (iter == g_batch_renderer_descriptors.end())
        return false;
    else
        iter->second = r;
        return true;
}

void batch_renderer_add_descriptor(uint64_t id, const BatchRenderer& r)
{
    g_batch_renderer_descriptors[id] = r;
}

void batch_renderer_rem_descriptor(uint64_t id)
{
    g_batch_renderer_descriptors.erase(id);
}

} // namespace wp

uint64_t wp_batch_renderer_create_host(
    int img_width,
    int img_height,
    int nworld,
    int ncam,
    float fov_rad,
    int ngeom,
    wp::array_t<int> geom_type,
    wp::array_t<wp::vec3> geom_size)
{
    (void)geom_type; (void)geom_size;
    BatchRenderer* r = new BatchRenderer(img_width, img_height, nworld, ncam, fov_rad, ngeom, nullptr);
    return (uint64_t)r;
}

void wp_batch_renderer_destroy_host(uint64_t id)
{
    BatchRenderer* r = (BatchRenderer*)(id);
    delete r;
}

void wp_batch_renderer_render_host(uint64_t id, wp::array_t<wp::vec3> cam_xpos, wp::array_t<wp::mat33> cam_xmat, wp::array_t<wp::vec3> geom_xpos, wp::array_t<wp::mat33> geom_xmat)
{
    BatchRenderer* r = (BatchRenderer*)(id);
    // TODO: implement host-side rendering
    (void)cam_xpos; (void)cam_xmat; (void)geom_xpos; (void)geom_xmat; (void)r;
}

#if !WP_ENABLE_CUDA
WP_API uint64_t wp_batch_renderer_create_device(void* context, int img_width, int img_height) { return 0; }
WP_API void wp_batch_renderer_destroy_device(uint64_t id) {}
WP_API void wp_batch_renderer_render_device(uint64_t id) {}
#endif // !WP_ENABLE_CUDA
