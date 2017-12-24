#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

#include "arithmetic/mi_cuda_math.h"
#include "mi_cuda_vr.h"

__device__ float ray_intersext_rectangle(float3 ray_start, float3 ray_dir, Rectangle& rect, float3* out) {
    return ray_intersext_rectangle(ray_start, ray_dir, rect.p0, rect.p1, rect.p2, rect.p3, out);
}

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
    if (x != 512 || y != 512) {
        return;
    }
    float ndc_x = ( x / (float)viewport.width - 0.5) * 2.0;
    float ndc_y = ( (viewport.height - y) / (float)viewport.height - 0.5 ) * 2.0;
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

        Rectangle(p001,p101,p111,p011,
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

    float dis0 = ray_intersext_rectangle(ray_start, ray_dir, rects[0], &hit_pos);
    float dis1 = ray_intersext_rectangle(ray_start, ray_dir, rects[1], &hit_pos);
    float dis2 = ray_intersext_rectangle(ray_start, ray_dir, rects[2], &hit_pos);
    float dis3 = ray_intersext_rectangle(ray_start, ray_dir, rects[3], &hit_pos);
    float dis4 = ray_intersext_rectangle(ray_start, ray_dir, rects[4], &hit_pos);
    float dis5 = ray_intersext_rectangle(ray_start, ray_dir, rects[5], &hit_pos);

    for (int i = 0; i < 6; ++i) {
        dis = ray_intersext_rectangle(ray_start, ray_dir, rects[i], &hit_pos);
        if (dis < min_dis) {
            min_dis = dis;
            hit = i;
        }
    }

    if (min_dis > -INF) {
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

extern "C" void init_navigator() {

}

extern "C" void ray_tracing(Viewport& viewport, int width, int height, mat4& mat_viewmodel, mat4& mat_projection_inv, unsigned char* result) {
#define BLOCK_SIZE 16
    CHECK_CUDA_ERROR;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(viewport.width / BLOCK_SIZE, viewport.height / BLOCK_SIZE);
    kernel_ray_tracing <<<grid_size, block_size >>> (viewport, width, height, mat_viewmodel, mat_projection_inv, result);
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;
    
}