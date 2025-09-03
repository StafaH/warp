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

#include "warp.h"
#include "cuda_util.h"
#include "raytrace.h"

namespace wp
{


__global__ void build_geoms_kernel(
    wp::Geom* __restrict__ geoms,
    const int ngeom,
    const int nworld,
    wp::array_t<int> geom_type,
    wp::array_t<wp::vec3> geom_size)
{
    const int idx = blockDim.x*blockIdx.x + threadIdx.x;
    const int total = nworld * ngeom;
    if (idx >= total)
        return;

    const int w = wp::floordiv(idx, ngeom);
    const int g = idx % ngeom;

    const int t = wp::index(geom_type, g);
    const wp::vec3 s = wp::index(geom_size, w, g);
    geoms[idx] = wp::Geom(t, s);
}

__global__ void render_megakernel(
    int n,
    const int nworld,
    const int ncam,
    const int img_width,
    const int img_height,
    const float fov_rad,
    const wp::vec3* cam_xpos,
    const wp::mat33* cam_xmat,
    const wp::vec3* geom_xpos,   // [nworld * ngeom]
    const wp::mat33* geom_xmat,  // [nworld * ngeom]
    const wp::Geom* geoms,       // [nworld * ngeom]
    const int ngeom)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index >= n) return;

    const int view_index_offset = wp::floordiv(index, img_width * img_height);
    const int pixel_index = index % (img_width * img_height);

    const int px = pixel_index % img_width;
    const int py = wp::floordiv(pixel_index, img_width);

    for (int i = 0; i < 8; i++)
    {
        const int view_index = view_index_offset + i;
        const int world_index = wp::floordiv(view_index, ncam);
        const int cam_index = view_index % ncam;

        wp::vec3 out_ray_dir;
        
        wp::compute_camera_ray(
            img_width,
            img_height,
            fov_rad,
            px,
            py,
            cam_xmat[(world_index * ncam) + cam_index],
            out_ray_dir);

        // noop: geom passthrough currently unused, reserved for future raycast
        (void)geom_xpos;
        (void)geom_xmat;
        (void)geoms;
        (void)ngeom;
    }
}

} // namespace wp


uint64_t wp_batch_renderer_create_device(
    void* context,
    int img_width,
    int img_height,
    int nworld,
    int ncam,
    float fov_rad,
    int ngeom,
    wp::array_t<int> geom_type,
    wp::array_t<wp::vec3> geom_size)
{
    ContextGuard guard(context);
    wp::BatchRenderer batchrenderer_device_on_host(
        img_width,
        img_height,
        nworld,
        ncam,
        fov_rad,
        ngeom,
        nullptr
    );
    // Allocate device-side renderer
    wp::BatchRenderer* batchrenderer_device_ptr = (wp::BatchRenderer*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BatchRenderer));

    // Allocate and build geoms (batched across worlds)
    const int total_geoms = nworld * ngeom;
    wp::Geom* geoms_dev = nullptr;
    if (total_geoms > 0)
    {
        geoms_dev = (wp::Geom*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::Geom) * (size_t)total_geoms);
        wp_launch_device(
            WP_CURRENT_CONTEXT,
            wp::build_geoms_kernel,
            total_geoms,
            ( geoms_dev, ngeom, nworld, geom_type, geom_size )
        );
    }

    // Publish geoms pointer into device struct and host descriptor
    batchrenderer_device_on_host.geoms = geoms_dev;

    // Upload renderer struct
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, batchrenderer_device_ptr, &batchrenderer_device_on_host, sizeof(wp::BatchRenderer));

    // Register descriptor (host-side mirror)
    uint64_t batchrenderer_id = (uint64_t)batchrenderer_device_ptr;
    wp::batch_renderer_add_descriptor(batchrenderer_id, batchrenderer_device_on_host);
    return batchrenderer_id;
}

void wp_batch_renderer_destroy_device(uint64_t id)
{
    wp::BatchRenderer r;
    if (wp::batch_renderer_get_descriptor(id, r))
    {
        ContextGuard guard(r.context);

        if (r.geoms)
        {
            wp_free_device(WP_CURRENT_CONTEXT, (void*)r.geoms);
        }
        wp_free_device(WP_CURRENT_CONTEXT, (wp::BatchRenderer*)id);

        wp::batch_renderer_rem_descriptor(id);
    }
}

void wp_batch_renderer_render_device(
    uint64_t id,
    wp::array_t<wp::vec3> cam_xpos,
    wp::array_t<wp::mat33> cam_xmat,
    wp::array_t<wp::vec3> geom_xpos,
    wp::array_t<wp::mat33> geom_xmat)
{
    ContextGuard guard(WP_CURRENT_CONTEXT);

    wp::BatchRenderer r;

    if (batch_renderer_get_descriptor(id, r))
    {
        int total_rays = r.nworld * r.ncam * r.img_width * r.img_height;
        int threads = wp::floordiv(total_rays, 8) + 1;
    
        wp_launch_device(
            WP_CURRENT_CONTEXT,
            wp::render_megakernel,
            threads,
            ( // arguments
                total_rays,
                r.nworld,
                r.ncam,
                r.img_width,
                r.img_height,
                r.fov_rad,
                cam_xpos,
                cam_xmat,
                geom_xpos,
                geom_xmat,
                r.geoms,
                r.ngeom
            ));
    }
    else
    {
        fprintf(stderr, "The batch renderer id provided to wp_batch_renderer_render_device is not valid!\n");
        return;
    }
}
