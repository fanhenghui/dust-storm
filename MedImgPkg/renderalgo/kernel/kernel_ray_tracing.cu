#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_functions.h>

#include "arithmetic/mi_cuda_graphic.h"

inline __device__ float4 blend(float4 src, float4 dst) {
    return make_float4(
        dst.w * dst.x + (1 - dst.w)*src.w * src.x,
        dst.w * dst.y + (1 - dst.w)*src.w * src.y,
        dst.w * dst.z + (1 - dst.w)*src.w * src.z,
        dst.w + (1 - dst.w)*src.w);
}

extern __shared__ float s_array[];//vertex(f3) , tex_coordinate(f2)
__global__ void kernel_ray_tracing_quad_vertex_mapping_shared_memory(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, float2* __restrict__ tex_coordinate, cudaTextureObject_t mapping_tex, cudaSurfaceObject_t canvas_surf,  bool is_blend) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    bool jump = false;
    if (x > viewport.width || y > viewport.height) {
        jump = true;
    }

    float3 *s_vertex = (float3*)s_array;
    float2 *s_tex_coordinate = (float2*)(&(s_array[vertex_count * 3]));

    //note: this kernel is for navigator's simple ray tracing(common's version should calcute fill shared memory's step)
    uint s_id = threadIdx.y*blockDim.x + threadIdx.x;
    if (s_id < vertex_count) {
        s_vertex[s_id] = mat_viewmodel*vertex[s_id];
        s_tex_coordinate[s_id] = tex_coordinate[s_id];
    }
    __syncthreads();

    if (jump) {
        return;
    }

    float ndc_x = (x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
    ray_start = mat_projection_inv*ray_start;
    ray_end = mat_projection_inv*ray_end;
    float3 ray_dir = ray_end - ray_start;
    ray_dir = normalize(ray_dir);

    int quad_count = vertex_count / 4;
    float3 p0, p1, p2, p3;
    float2 t0, t1, t2;
    float3 uvw, cur_uvw;
    int hit = -1;
    int tri_type = -1;
    int cur_tri_type;
    float min_dis = INF;
    float cur_dis = 0;
    float3 out;
    for (int i = 0; i < quad_count; ++i) {
        p0 = s_vertex[i * 4];
        p1 = s_vertex[i * 4 + 1];
        p2 = s_vertex[i * 4 + 2];
        p3 = s_vertex[i * 4 + 3];

        //to remove a gap between two triangles
        cur_dis = ray_intersect_rectangle(ray_start, ray_dir, p0, p1, p2, p3, &cur_uvw, &out, cur_tri_type);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            tri_type = cur_tri_type;
        }
    }

    if (hit == -1) {
        if (!is_blend) {
            uchar4 rgba = make_uchar4(0,0,0,0);
            surf2Dwrite(rgba, canvas_surf, (viewport.x + x) * 4, viewport.y + y);
        }
        return;
    }

    //triangle interpolate
    if (0 == tri_type) {
        t0 = s_tex_coordinate[hit * 4];
        t1 = s_tex_coordinate[hit * 4 + 1];
        t2 = s_tex_coordinate[hit * 4 + 2];
    }
    else {
        t0 = s_tex_coordinate[hit * 4];
        t1 = s_tex_coordinate[hit * 4 + 2];
        t2 = s_tex_coordinate[hit * 4 + 3];
    }
    float2 tex_coord = t0*uvw.x + t1*uvw.y + t2*uvw.z;
    float4 color_dst = tex2D<float4>(mapping_tex, tex_coord.x, tex_coord.y);

    float4 color;
    if (is_blend) {
        float num_255_r = 0.003921f;
        uchar4 color_src8;
        surf2Dread(&color_src8, canvas_surf, (viewport.x + x) * 4, viewport.y + y);
        float4 color_src = make_float4(
            color_src8.x * num_255_r,
            color_src8.y * num_255_r,
            color_src8.z * num_255_r,
            color_src8.w * num_255_r);
        color = blend(color_src, color_dst);
    }
    else {
        color = color_dst;
    }
    clamp(color, 0.0f, 1.0f);
     
    uchar4 rgba = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
    surf2Dwrite(rgba, canvas_surf, (viewport.x + x) * 4, viewport.y + y);
}

extern "C"
cudaError_t ray_tracing_quad_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv, mat4 mat_mvp,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaTextureObject_t mapping_tex, cudaSurfaceObject_t canvas_surf, bool is_blend)
{
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    if (grid.x * BLOCK_SIZE != viewport.width) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZE != viewport.height) {
        grid.y += 1;
    }
    kernel_ray_tracing_quad_vertex_mapping_shared_memory << <grid, block, vertex_count * 20 >> > (viewport, width, height, 
        mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_tex_coordinate, mapping_tex, canvas_surf, is_blend);

    return cudaThreadSynchronize();
}