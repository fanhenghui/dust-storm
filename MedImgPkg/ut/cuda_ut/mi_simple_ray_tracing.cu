#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

#include "arithmetic/mi_cuda_math.h"
#include "mi_cuda_graphic.h"

__device__ float ray_intersect_rectangle(float3 ray_start, float3 ray_dir, Rectangle& rect, float3* out) {
    return ray_intersect_rectangle(ray_start, ray_dir, rect.p0, rect.p1, rect.p2, rect.p3, out);
}

__global__ void kernel_ray_tracing_element_vertex_color(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* vertex, int ele_count, int* element, float4* color, unsigned char* result) {

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    uint idx = y*width + x;

    float ndc_x = (x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
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
    float cur_dis = 0;
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
    clamp(p_color, 0, 1);
    
    result[idx * 4] = p_color.x*255;
    result[idx * 4 + 1] = p_color.y * 255;
    result[idx * 4 + 2] = p_color.z * 255;
    result[idx * 4 + 3] = 255;   
}

__global__ void kernel_ray_tracing_element_vertex_mapping(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* vertex, int ele_count, int* element, float2* tex_coordinate, cudaTextureObject_t tex, unsigned char* result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}

    uint idx = y*width + x;

    float ndc_x = (x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
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
    int vertex_count, float3* vertex, float4* color, unsigned char* result) {

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    uint idx = y*width + x;

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
    int vertex_count, float3* vertex, float2* tex_coordinate, cudaTextureObject_t tex, unsigned char* result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}

    uint idx = y*width + x;

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
    int vertex_count, float3* vertex, float2* tex_coordinate, cudaTextureObject_t tex, unsigned char* result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }

    //if (x != 255 || y != 255) {
    //    return;
    //}

    uint idx = y*width + x;

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

extern __shared__ float s_array[];//vertex(f3) , tex_coordinate(f2)

__global__ void kernel_ray_tracing_quad_vertex_mapping_ext(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* vertex, float2* tex_coordinate, cudaTextureObject_t tex, unsigned char* result) {
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

    uint idx = y*width + x;

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
    float min_dis = INF;
    float cur_dis = 0;
    float3 out;
    for (int i = 0; i < quad_count; ++i) {
        p0 = s_vertex[i * 4];
        p1 = s_vertex[i * 4 + 1];
        p2 = s_vertex[i * 4 + 2];
        p3 = s_vertex[i * 4 + 3];

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p1, p2, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            tri_type = 0;
            continue;
        }

        cur_dis = ray_intersect_triangle(ray_start, ray_dir, p0, p2, p3, &cur_uvw, &out);
        if (cur_dis > -INF && cur_dis < min_dis) {
            min_dis = cur_dis;
            hit = i;
            uvw = cur_uvw;
            tri_type = 1;
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
    float4 color = tex2D<float4>(tex, tex_coord.x, tex_coord.y);
    clamp(color, 0, 1);

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
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex)
{
    const int BLOCK_SIZE = 16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    //kernel_ray_tracing_quad_vertex_mapping << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result);
    kernel_ray_tracing_quad_vertex_mapping_ext << <grid_size, block_size, vertex_count*20 >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, d_tex_coordinate, mapping_tex.cuda_tex_obj, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;
}