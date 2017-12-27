#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

#include "mi_cuda_graphic.h"
#include "mi_cuda_vr_common.h"

__global__ void kernel_ray_tracing_element_vertex_color(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, int ele_count, int* __restrict__  element, float4* __restrict__  color, unsigned char* __restrict__ result) {

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    uint idx = (y + viewport.y)*width + x + viewport.x;

    float ndc_x = (x / (float)viewport.width - 0.5f) * 2.0f;
    float ndc_y = (y / (float)viewport.height - 0.5f) * 2.0f;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0f);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0f);
    ray_start = mat_projection_inv*ray_start;
    ray_end = mat_projection_inv*ray_end;
    float3 ray_dir = ray_end - ray_start;
    ray_dir = normalize(ray_dir);


    int tri_count = ele_count / 3;
    float3 p0, p1, p2;
    float4 c0, c1, c2;
    float3 uvw, cur_uvw;
    int ele0, ele1, ele2;
    int hit = -1;
    float min_dis = INF;
    float cur_dis = 0.0f;
    float3 out;
    for (int i = 0; i < tri_count; ++i) {
        ele0 = element[i * 3];
        ele1 = element[i * 3 + 1];
        ele2 = element[i * 3 + 2];
        p0 = mat_viewmodel*vertex[ele0];
        p1 = mat_viewmodel*vertex[ele1];
        p2 = mat_viewmodel*vertex[ele2];

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0,p1,p2, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            c0 = color[ele0];
            c1 = color[ele1];
            c2 = color[ele2];
        }
    }

    if (hit == -1) {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
        return;
    }

    //triangle interpolate
    float4 p_color = c0*uvw.x + c1*uvw.y + c2*uvw.z;
    clamp(p_color, 0.0f, 1.0f);
    
    result[idx * 4] = p_color.x*255;
    result[idx * 4 + 1] = p_color.y * 255;
    result[idx * 4 + 2] = p_color.z * 255;
    result[idx * 4 + 3] = 255;   
}

__global__ void kernel_ray_tracing_element_vertex_mapping(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, int ele_count, int* __restrict__ element, float2* __restrict__ tex_coordinate, cudaTextureObject_t tex, unsigned char* __restrict__ result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}

    uint idx = (y+viewport.y)*width + x+viewport.x;

    float ndc_x = (x / (float)viewport.width - 0.5f) * 2.0f;
    float ndc_y = (y / (float)viewport.height - 0.5f) * 2.0f;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0f);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0f);
    ray_start = mat_projection_inv*ray_start;
    ray_end = mat_projection_inv*ray_end;
    float3 ray_dir = ray_end - ray_start;
    ray_dir = normalize(ray_dir);


    int tri_count = ele_count / 3;
    float3 p0, p1, p2;
    float2 t0, t1, t2;
    float3 uvw, cur_uvw;
    int ele0, ele1, ele2;
    int hit = -1;
    float min_dis = INF;
    float cur_dis = 0;
    float3 out;
    for (int i = 0; i < tri_count; ++i) {
        ele0 = element[i * 3];
        ele1 = element[i * 3 + 1];
        ele2 = element[i * 3 + 2];
        p0 = mat_viewmodel*vertex[ele0];
        p1 = mat_viewmodel*vertex[ele1];
        p2 = mat_viewmodel*vertex[ele2];

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p1, p2, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            t0 = tex_coordinate[ele0];
            t1 = tex_coordinate[ele1];
            t2 = tex_coordinate[ele2];
        }
    }

    if (hit == -1) {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
        return;
    }

    //triangle interpolate
    float2 tex_coord = t0*uvw.x + t1*uvw.y + t2*uvw.z;
    float4 color = tex2D<float4>(tex, tex_coord.x, tex_coord.y);
    clamp(color, 0, 1);

    result[idx * 4] = color.x * 255;
    result[idx * 4 + 1] = color.y * 255;
    result[idx * 4 + 2] = color.z * 255;
    result[idx * 4 + 3] = 255;

}

__global__ void kernel_ray_tracing_triangle_vertex_color(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, float4* __restrict__ color, unsigned char* __restrict__ result) {

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    uint idx = (y + viewport.y)*width + x + viewport.x;

    float ndc_x = (x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
    ray_start = mat_projection_inv*ray_start;
    ray_end = mat_projection_inv*ray_end;
    float3 ray_dir = ray_end - ray_start;
    ray_dir = normalize(ray_dir);


    int tri_count = vertex_count / 3;
    float3 p0, p1, p2;
    float4 c0, c1, c2;
    float3 uvw, cur_uvw;
    int hit = -1;
    float min_dis = INF;
    float cur_dis = 0;
    float3 out;
    for (int i = 0; i < tri_count; ++i) {
        p0 = mat_viewmodel*vertex[i * 3];
        p1 = mat_viewmodel*vertex[i * 3 + 1];
        p2 = mat_viewmodel*vertex[i * 3 + 2];

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p1, p2, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            c0 = color[i * 3];
            c1 = color[i * 3 + 1];
            c2 = color[i * 3 + 2];
        }
    }

    if (hit == -1) {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
        return;
    }

    //triangle interpolate
    float4 p_color = c0*uvw.x + c1*uvw.y + c2*uvw.z;
    clamp(p_color, 0, 1);

    result[idx * 4] = p_color.x * 255;
    result[idx * 4 + 1] = p_color.y * 255;
    result[idx * 4 + 2] = p_color.z * 255;
    result[idx * 4 + 3] = 255;
}

__global__ void kernel_ray_tracing_triangle_vertex_mapping(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, float2* __restrict__ tex_coordinate, cudaTextureObject_t tex, unsigned char* __restrict__ result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}

    uint idx = (y + viewport.y)*width + x + viewport.x;

    float ndc_x = (x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
    ray_start = mat_projection_inv*ray_start;
    ray_end = mat_projection_inv*ray_end;
    float3 ray_dir = ray_end - ray_start;
    ray_dir = normalize(ray_dir);


    int tri_count = vertex_count / 3;
    float3 p0, p1, p2;
    float2 t0, t1, t2;
    float3 uvw, cur_uvw;
    int hit = -1;
    float min_dis = INF;
    float cur_dis = 0;
    float3 out;
    for (int i = 0; i < tri_count; ++i) {
        p0 = mat_viewmodel*vertex[i * 3];
        p1 = mat_viewmodel*vertex[i * 3 + 1];
        p2 = mat_viewmodel*vertex[i * 3 + 2];

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0,p1,p2, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            t0 = tex_coordinate[i * 3];
            t1 = tex_coordinate[i * 3 + 1];
            t2 = tex_coordinate[i * 3 + 2];
        }
    }

    if (hit == -1) {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
        return;
    }

    //triangle interpolate
    float2 tex_coord = t0*uvw.x + t1*uvw.y + t2*uvw.z;
    float4 color = tex2D<float4>(tex, tex_coord.x, tex_coord.y);
    clamp(color, 0, 1);
    
    result[idx * 4] = color.x*255;
    result[idx * 4 + 1] = color.y * 255;
    result[idx * 4 + 2] = color.z * 255;
    result[idx * 4 + 3] = 255;   
}

__global__ void kernel_ray_tracing_quad_vertex_mapping(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, float2* __restrict__ tex_coordinate, cudaTextureObject_t tex, unsigned char* __restrict__ result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}

    uint idx = (y + viewport.y)*width + x + viewport.x;

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
    float min_dis = INF;
    float cur_dis = 0;
    float3 out;
    for (int i = 0; i < quad_count; ++i) {
        p0 = mat_viewmodel*vertex[i * 4];
        p1 = mat_viewmodel*vertex[i * 4 + 1];
        p2 = mat_viewmodel*vertex[i * 4 + 2];
        p3 = mat_viewmodel*vertex[i * 4 + 3];

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p1, p2, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            t0 = tex_coordinate[i * 4];
            t1 = tex_coordinate[i * 4 + 1];
            t2 = tex_coordinate[i * 4 + 2];
            continue;
        }

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p2, p3, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            t0 = tex_coordinate[i * 4];
            t1 = tex_coordinate[i * 4 + 2];
            t2 = tex_coordinate[i * 4 + 3];
        }
    }

    if (hit == -1) {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
        return;
    }

    //triangle interpolate
    float2 tex_coord = t0*uvw.x + t1*uvw.y + t2*uvw.z;
    float4 color = tex2D<float4>(tex, tex_coord.x, tex_coord.y);
    clamp(color, 0, 1);

    result[idx * 4] = color.x * 255;
    result[idx * 4 + 1] = color.y * 255;
    result[idx * 4 + 2] = color.z * 255;
    result[idx * 4 + 3] = 255;
}


inline __device__ float4 blend(float4 src, float4 dst) {
    return make_float4(
        dst.w * dst.x + (1 - dst.w)*src.w * src.x,
        dst.w * dst.y + (1 - dst.w)*src.w * src.y,
        dst.w * dst.z + (1 - dst.w)*src.w * src.z,
        dst.w + (1 - dst.w)*src.w);
}

extern __shared__ float s_array[];//vertex(f3) , tex_coordinate(f2)

//use shared memory to replace global memory
__global__ void kernel_ray_tracing_quad_vertex_mapping_shared_memory(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* __restrict__ vertex, float2* __restrict__ tex_coordinate, cudaTextureObject_t tex, unsigned char* __restrict__ result, bool is_blend) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    bool jump = false;
    if (x > viewport.width || y > viewport.height) {
        jump = true;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}


    float3 *s_vertex = (float3*)s_array;
    float2 *s_tex_coordinate = (float2*)(&(s_array[vertex_count * 3]));

    uint s_id = threadIdx.y*blockDim.x + threadIdx.x;

    // TODO when vertex is more than blockDim.x*blockDim.y
    /*uint s_all = blockDim.x*blockDim.y;
    uint s_step = vertex_count / s_all;
    if (s_step == 0) {
        s_step = 1;
    }*/

    //!!slower a little
    /*if (s_id < vertex_count) {
        s_vertex[s_id] = mat_viewmodel*vertex[s_id];
    } else if(s_id < vertex_count*2) {
        s_tex_coordinate[s_id - vertex_count] = tex_coordinate[s_id - vertex_count];
    }*/

    if (s_id < vertex_count) {
        s_vertex[s_id] = mat_viewmodel*vertex[s_id];
        s_tex_coordinate[s_id] = tex_coordinate[s_id];
    }

    __syncthreads(); //await to fill all vertex & texture coordinate to shared mamory

    if (jump) {
        return;
    }

    uint idx = (y + viewport.y)*width + x + viewport.x;

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

        //cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p1, p2, &cur_uvw, &out);
        //if (cur_dis > -INF && cur_dis < min_dis) {
        //    min_dis = cur_dis;
        //    hit = i;
        //    uvw = cur_uvw;
        //    tri_type = 0;
        //    continue;
        //}

        //cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p2, p3, &cur_uvw2, &out);
        //if (cur_dis > -INF && cur_dis < min_dis) {
        //    min_dis = cur_dis;
        //    hit = i;
        //    uvw = cur_uvw2;
        //    tri_type = 1;
        //}

        
        //to remove a gap between two triangles
        cur_dis = ray_intersect_rectangle(ray_start, ray_dir, p0, p1, p2, p3, &cur_uvw, &out, cur_tri_type);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            tri_type = cur_tri_type;
        }
    }

    //__syncthreads(); no need

    if (hit == -1) {
        if (!is_blend) {
            result[idx * 4] = 0;
            result[idx * 4 + 1] = 0;
            result[idx * 4 + 2] = 0;
            result[idx * 4 + 3] = 0;
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
    float4 color_dst = tex2D<float4>(tex, tex_coord.x, tex_coord.y);

    float4 color;
    if (is_blend) {
        float4 color_src = make_float4(result[idx * 4] * 0.003921f, result[idx * 4 + 1] * 0.003921f, result[idx * 4 + 2] * 0.003921f, result[idx * 4 + 3] * 0.003921f);
        color = blend(color_src, color_dst);
    }
    else {
        color = color_dst;
    }
    clamp(color, 0.0f, 1.0f);
    
    result[idx * 4] = color.x * 255;
    result[idx * 4 + 1] = color.y * 255;
    result[idx * 4 + 2] = color.z * 255;
    result[idx * 4 + 3] = 255;
}

//use shared memory and resterization (TODO with bug !! Z-fighting ??)
__global__ void kernel_ray_tracing_quad_vertex_mapping_resterization(Viewport viewport, int width, int height, mat4 matmvp,
    int vertex_count, float3* __restrict__ vertex, float2* __restrict__ tex_coordinate, cudaTextureObject_t tex, unsigned char* __restrict__ result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const int quad_count = vertex_count / 4;
    //const int triangle_count = quad_count*2;

    bool jump = false;
    if (x > viewport.width || y > viewport.height) {
        jump = true;
    }

    //one thread process one quad
    //float3 *s_vertex = (float3*)s_array;
    //float2 *s_tex_coordinate = (float2*)(&(s_array[vertex_count * 3]));
    //float4 *s_triangle_cache = (float4*)(&(s_array[vertex_count * 5]));

    //uint s_id = threadIdx.y*blockDim.x + threadIdx.x;

    //if (s_id < triangle_count) {
    //    uint s_id0 = s_id / 2;
    //    if (s_id % 2 == 0) {
    //        //4 vertex
    //        s_vertex[s_id0 * 4] = matmvp*vertex[s_id0 * 4];
    //        s_vertex[s_id0 * 4 + 1] = matmvp*vertex[s_id0 * 4 + 1];
    //        s_vertex[s_id0 * 4 + 2] = matmvp*vertex[s_id0 * 4 + 2];
    //        s_vertex[s_id0 * 4 + 3] = matmvp*vertex[s_id0 * 4 + 3];
    //        //4 texture coordinate 
    //        s_tex_coordinate[s_id0 * 4] = tex_coordinate[s_id0 * 4];
    //        s_tex_coordinate[s_id0 * 4 + 1] = tex_coordinate[s_id0 * 4 + 1];
    //        s_tex_coordinate[s_id0 * 4 + 2] = tex_coordinate[s_id0 * 4 + 2];
    //        s_tex_coordinate[s_id0 * 4 + 3] = tex_coordinate[s_id0 * 4 + 3];
    //        // calclate triangle p0p1p2
    //        float4 cache_param;
    //        triangle_barycentric_cache_2d(make_float2(s_vertex[s_id0 * 4]), make_float2(s_vertex[s_id0 * 4 + 1]), make_float2(s_vertex[s_id0 * 4 + 2]), cache_param);
    //        s_triangle_cache[s_id] = cache_param;
    //    } else {
    //        // calclate triangle p0p2p3

    //        float4 cache_param;
    //        triangle_barycentric_cache_2d(make_float2(vertex[s_id0 * 4]), make_float2(vertex[s_id0 * 4 + 2]), make_float2(vertex[s_id0 * 4 +3]), cache_param);
    //        s_triangle_cache[s_id] = cache_param;
    //    }
    //}


    float3 *s_vertex = (float3*)s_array;
    float2 *s_tex_coordinate = (float2*)(&(s_array[vertex_count * 3]));
    uint s_id = threadIdx.y*blockDim.x + threadIdx.x;
    if (s_id < vertex_count) {
        s_vertex[s_id] = matmvp*vertex[s_id];
        s_tex_coordinate[s_id] = tex_coordinate[s_id];
    }

    __syncthreads(); //await to fill all vertex & texture coordinate to shared mamory

    if (jump) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}


    uint idx = (y + viewport.y)*width + x + viewport.x;
    float ndc_x = (x / (float)viewport.width - 0.5f) * 2.0f;
    float ndc_y = (y / (float)viewport.height - 0.5f) * 2.0f;
    float2 p = make_float2(ndc_x, ndc_y);
    float3 uvw, cur_uvw;
    int hit = -1;
    int tri_type = -1;
    float cur_z;
    float z_min = INF;
    float3 p0, p1, p2, p3;
    for (int i = 0; i < quad_count; ++i) {
        p0 = s_vertex[i * 4];
        p1 = s_vertex[i * 4 + 1];
        p2 = s_vertex[i * 4 + 2];
        p3 = s_vertex[i * 4 + 3];
        
        //triangle p0p1p2        
        //triangle_barycentric_ext_2d(make_float2(p0), make_float2(p1), make_float2(p2), p, s_triangle_cache[i*2], cur_uvw);
        if (triangle_barycentric_2d(make_float2(p0), make_float2(p1), make_float2(p2), p, cur_uvw)) {
            cur_z = p0.z * uvw.x + p1.z * uvw.y + p2.z * uvw.z;
            if (cur_z < z_min) {
                uvw = cur_uvw;
                hit = i;
                tri_type = 0;
                z_min = cur_z;
            }
        }

        //triangle p0p2p3
        //triangle_barycentric_ext_2d(make_float2(p0), make_float2(p2), make_float2(p3), p, s_triangle_cache[i * 2 + 1], cur_uvw)
        if (triangle_barycentric_2d(make_float2(p0), make_float2(p2), make_float2(p3), p, cur_uvw)) {
            cur_z = p0.z * uvw.x + p2.z * uvw.y + p3.z * uvw.z;
            if (cur_z < z_min) {
                uvw = cur_uvw;
                hit = i;
                tri_type = 1;
                z_min = cur_z;
            }
        }
    }

    //__syncthreads(); no need

    if (hit == -1) {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
        return;
    } 
    if (0 == tri_type) {
        result[idx * 4 + 0] = 255;
        result[idx * 4 + 1] = 255;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 255;

    }
    else {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 255;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 255;
    }
    return;

    //triangle interpolate
    float2 t0,t1,t2;
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
    float4 color = tex2D<float4>(tex, tex_coord.x, tex_coord.y);
    clamp(color, 0.0f, 1.0f);

    result[idx * 4] = color.x * 255;
    result[idx * 4 + 1] = color.y * 255;
    result[idx * 4 + 2] = color.z * 255;
    result[idx * 4 + 3] = 255;
}


//------------------------------------------------------------------------------------//

extern "C"
void ray_tracing_element_vertex_color(Viewport viewport, int width, int height, 
                              mat4 mat_viewmodel, mat4 mat_projection_inv,
                              int vertex_count, float3* d_vertex, int ele_count, int* d_element, float4* d_color, 
                              unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex)
{
    const int BLOCK_SIZE=16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing_element_vertex_color << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, ele_count, d_element, d_color, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;
}

extern "C"
void ray_tracing_element_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, int ele_count, int* d_element, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex)
{
    const int BLOCK_SIZE = 16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing_element_vertex_mapping << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, ele_count, d_element, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;
}

extern "C"
void ray_tracing_triangle_vertex_color(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, float4* d_color,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex) 
{
    const int BLOCK_SIZE = 16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing_triangle_vertex_color << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_color, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;
}

extern "C"
void ray_tracing_triangle_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex)
{
    const int BLOCK_SIZE = 16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing_triangle_vertex_mapping << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;
}

extern "C"
void ray_tracing_quad_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv, mat4 mat_mvp,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex, bool is_blend)
{
    const int BLOCK_SIZE = 16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    //kernel_ray_tracing_quad_vertex_mapping << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result);
    
    kernel_ray_tracing_quad_vertex_mapping_shared_memory << <grid_size, block_size, vertex_count*20 >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result, is_blend);

    //kernel_ray_tracing_quad_vertex_mapping_resterization << <grid_size, block_size, vertex_count * 20  >> > (viewport, width, height, mat_mvp, vertex_count, d_vertex, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;
}