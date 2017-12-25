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

__global__ void kernel_ray_tracing_vertex_color(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
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

//just support triangle
//__global__ void kernel_ray_tracing_vertex_color(Viewport viewport, int width, int height, mat4 mat_viewmodel, mat4 mat_projection_inv,
//    uint vertex_count, float3* vertex, uint ele_count,  uint* element, float2* texture_coordinate, unsigned char* result) {
//    uint x = blockIdx.x * blockDim.x + threadIdx.x;
//    uint y = blockIdx.y * blockDim.y + threadIdx.y;   
//
//    if (x > viewport.width || y > viewport.height) {
//        return;
//    }
//    uint idx = y*width + x;
//
//    float ndc_x = (x / (float)viewport.width - 0.5) * 2.0;
//    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
//    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
//    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
//    ray_start = mat_projection_inv*ray_start;
//    ray_end = mat_projection_inv*ray_end;
//    float3 ray_dir = ray_end - ray_start;
//    ray_dir = normalize(ray_dir);
//
//    
//    uint tri_count = ele_count/3;
//    float3 p0,p1,p2;
//    float4 c0,c1,c2;
//    uint ele0, ele1, ele2;
//    for (uint i = 0; i < tri_count; ++i) {
//        ele0 = element[i * 3];
//        ele1 = element[i * 3 + 1];
//        ele2 = element[i * 3 + 2];
//        p0 = vertex[ele0];
//        p1 = vertex[ele1];
//        p2 = vertex[ele2];
//        c0 = color[ele0];
//        c1 = color[ele1];
//        c2 = color[ele2];
//    }
//
//}



__global__ void kernel_ray_tracing(Viewport viewport, int width , int height, mat4 mat_viewmodel, mat4 mat_projection_inv, unsigned char* result ) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > viewport.width || y > viewport.height) {
        return;
    }
    uint idx = y*width + x;

    /*{
        result[idx * 4] = 255;
        result[idx * 4 + 1] = 255;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 255;
        return;
    }*/
    

    //hit 6 rectangle( easy to calcuate texture coordinate)
    //if (x != 512 || y != 512) {
    //    return;
    //}
    float ndc_x = ( x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = (y / (float)viewport.height - 0.5) * 2.0;
    float3 ray_start = make_float3(ndc_x, ndc_y, -1.0);
    float3 ray_end = make_float3(ndc_x, ndc_y, 1.0);
    ray_start = mat_projection_inv*ray_start;
    ray_end = mat_projection_inv*ray_end;
    float3 ray_dir = ray_end - ray_start;
    ray_dir = normalize(ray_dir);

    //navigator : center(0,0,0) length(1.2)
    float w = 0.6;
    float3 p000 = mat_viewmodel*make_float3(-w, -w, -w);
    float3 p001 = mat_viewmodel*make_float3(-w, -w, w);
    float3 p010 = mat_viewmodel*make_float3(-w, w, -w);
    float3 p011 = mat_viewmodel*make_float3(-w, w, w);
    float3 p100 = mat_viewmodel*make_float3(w, -w, -w);
    float3 p101 = mat_viewmodel*make_float3(w, -w, w);
    float3 p110 = mat_viewmodel*make_float3(w, w, -w);
    float3 p111 = mat_viewmodel*make_float3(w, w, w);

    //texture coordinate
    float x_step = 0.33333;
    float y_step = 0.5;

    Rectangle rects[6] = {
        Rectangle(p001,p101,p111,p011,
        make_float2(x_step * 2,y_step),make_float2(x_step * 3,y_step),make_float2(x_step * 3,0),make_float2(x_step * 2,0)),//head

        Rectangle(p000,p010,p110,p100,
        make_float2(x_step * 2,y_step * 2),make_float2(x_step * 2,y_step),make_float2(x_step * 3,y_step),make_float2(x_step * 3,y_step * 2)),//foot

        Rectangle(p000,p001,p011,p010,
        make_float2(x_step,0),make_float2(x_step,y_step),make_float2(0,y_step),make_float2(0,0)),//left

        Rectangle(p100,p110,p111,p101,
        make_float2(0,y_step),make_float2(x_step,y_step),make_float2(x_step,y_step * 2),make_float2(0,y_step * 2)),//right

        Rectangle(p010,p011,p111,p110,
        make_float2(x_step * 2,0),make_float2(x_step * 2,y_step),make_float2(x_step,y_step),make_float2(x_step,0)),//posterior

        Rectangle(p000,p100,p101,p001,
        make_float2(x_step,y_step),make_float2(x_step * 2,y_step),make_float2(x_step * 2,y_step * 2),make_float2(x_step,y_step * 2))//anterior
    };

    float min_dis = INF;
    float dis = 0;
    int hit = -1;
    float3 hit_pos;

    /*float dis0 = ray_intersect_rectangle(ray_start, ray_dir, rects[0], &hit_pos);
    float dis1 = ray_intersect_rectangle(ray_start, ray_dir, rects[1], &hit_pos);
    float dis2 = ray_intersect_rectangle(ray_start, ray_dir, rects[2], &hit_pos);
    float dis3 = ray_intersect_rectangle(ray_start, ray_dir, rects[3], &hit_pos);
    float dis4 = ray_intersect_rectangle(ray_start, ray_dir, rects[4], &hit_pos);
    float dis5 = ray_intersect_rectangle(ray_start, ray_dir, rects[5], &hit_pos);*/

    for (int i = 0; i < 6; ++i) {
        dis = ray_intersect_rectangle(ray_start, ray_dir, rects[i], &hit_pos);
        if (dis < min_dis && dis > -INF) {
            min_dis = dis;
            hit = i;
        }
    }

    if (hit != -1) {
        //use out to interpolate texture coordinate
        result[idx * 4 + 0] = 255;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 255;
    } else {
        result[idx * 4 + 0] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4 + 3] = 0;
    }
}

extern "C" void ray_tracing(Viewport& viewport, int width, int height, mat4& mat_viewmodel, mat4& mat_projection_inv, unsigned char* result, cudaGLTextureWriteOnly& cuda_tex) {
    const int BLOCK_SIZE = 16;
    CHECK_CUDA_ERROR;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing <<<grid_size, block_size >>> (viewport, width, height, mat_viewmodel, mat_projection_inv, result);
    
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;
    
    map_image(cuda_tex);

    CHECK_CUDA_ERROR;

    write_image(cuda_tex, result, width*height*4);

    CHECK_CUDA_ERROR;

    unmap_image(cuda_tex);

    CHECK_CUDA_ERROR;
}

extern "C"
void ray_tracing_vertex_color(Viewport viewport, int width, int height, 
                              mat4 mat_viewmodel, mat4 mat_projection_inv,
                              int vertex_count, float3* d_vertex, int ele_count, int* d_element, float4* d_color, 
                              unsigned char* d_result, cudaGLTextureWriteOnly& cuda_tex) 
{
    const int BLOCK_SIZE=16;
    CHECK_CUDA_ERROR;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing_vertex_color << <grid_size, block_size >> > (viewport, width, height, mat_viewmodel, mat_projection_inv, vertex_count, d_vertex, ele_count, d_element, d_color, d_result);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    map_image(cuda_tex);

    write_image(cuda_tex, d_result, width*height * 4);

    unmap_image(cuda_tex);

    CHECK_CUDA_ERROR;
}