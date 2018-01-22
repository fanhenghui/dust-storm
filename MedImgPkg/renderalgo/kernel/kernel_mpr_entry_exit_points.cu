#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_functions.h>

#include "arithmetic/mi_cuda_graphic.h"

inline __device__  float ray_intersect_brick(float3 init_pt, float3 brick_min, float3 brick_dim, float3 ray_dir, float* start_step, float* end_step) {
    float3 ray_r = 1.0f/ray_dir;
    float3 bottom = (brick_min - init_pt);
    float3 top = (brick_min + brick_dim - init_pt);
    float3 tbot = ray_r * bottom;
    float3 ttop = ray_r * top;

    //Adjust
    if (fabs(bottom.x) < EPSILON) {
        tbot.x = 0.0f;
    }
    if (fabs(bottom.y) < EPSILON) {
        tbot.y = 0.0f;
    }
    if (fabs(bottom.z) < EPSILON) {
        tbot.z = 0.0f;
    }

    if (fabs(top.x) < EPSILON) {
        ttop.x = 0.0f;
    }
    if (fabs(top.y) < EPSILON) {
        ttop.y = 0.0f;
    }
    if (fabs(top.z) < EPSILON) {
        ttop.z = 0.0f;
    }

    float3 tmin = fminf(tbot, ttop);
    float3 tmax = fmaxf(tbot, ttop);
    float tnear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float tfar = fminf(fminf(tmax.x, tmax.y), tmax.z);

    *start_step = tnear;
    *end_step = tfar;

    return tnear - *start_step;
}

inline __device__ bool outside(float3 pt , float3 bound) {
    if (pt.x < 0.0f || pt.y < 0.0f || pt.z < 0.0f) {
        return true;
    }
    if (pt.x > bound.x || pt.y > bound.y|| pt.z > bound.z) {
        return true;
    }
    return false;
}

__global__ void kernel_calculate_mpr_entry_exit_points(cudaSurfaceObject_t entry_surf, cudaSurfaceObject_t exit_surf, 
    int width, int height, mat4 mat_mvp_inv,  float3 volume_dim, float thickness, float3 ray_dir ) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width - 1 || y > height - 1) {
        return;
    }

    float ndc_x = (float(x) + 0.5f) / float(width);
    float ndc_y = (float(y) + 0.5f) / float(height);
    float3 pos_ndc = make_float3(ndc_x * 2.0f - 1.0f, ndc_y * 2.0f - 1.0f, 0.0f);
    float4 central4 = mat_mvp_inv * make_float4(pos_ndc, 1.0f);
    float3 central = make_float3(central4/ central4.w);

    float3 entry_point, exit_point;
    if (thickness <= 1.0f) {
        entry_point = central;
        exit_point = central + ray_dir * thickness;
    } else {
        entry_point = central - ray_dir * thickness * 0.5f;
        exit_point = central + ray_dir * thickness * 0.5f;
    }

    float entry_step = 0.0f;
    float exit_step = 0.0f;
    float3 entry_intersection = entry_point;
    float3 exit_intersection = exit_point;
    ray_intersect_brick(entry_point, make_float3(0.0f, 0.0f, 0.0f), volume_dim, ray_dir, &entry_step, &exit_step);

    //Entry point outside
    if (outside(entry_point, volume_dim - make_float3(1.0f))) {        
        if (entry_step >= exit_step || entry_step < 0 || entry_step > thickness)// check entry points in range of thickness and volume
        {
            exit_step = -1.0f;
            surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, -1.0f), entry_surf, x << 4, y);
            surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, -1.0f), exit_surf, x << 4, y);
            return;
        }
        entry_intersection = entry_point + entry_step * ray_dir;
    }

    //Exit point outside
    if (outside(exit_point, volume_dim - make_float3(1.0f)))
    {
        if (entry_step >= exit_step)
        {
            exit_step = -1.0f;
            surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, -1.0f), entry_surf, x << 4, y);
            surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, -1.0f), exit_surf, x << 4, y);
            return;
        }
        exit_intersection = entry_point + exit_step * ray_dir;
        if (thickness <= 1.0f)//forbid border exit_step=0 : entry == exit (ray direction is Nan)
        {
            exit_intersection = entry_point + thickness * ray_dir;
        }
    }

    surf2Dwrite(make_float4(entry_intersection, -1.0f), entry_surf, x << 4, y);
    surf2Dwrite(make_float4(exit_intersection, -1.0f), exit_surf, x << 4, y);
}

extern "C" 
cudaError_t calculate_mpr_entry_exit_points(cudaSurfaceObject_t entry_surf, cudaSurfaceObject_t exit_surf,
    int width, int height, mat4 mat_mvp_inv, float3 volume_dim, float thickness, float3 ray_dir) {
    
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    if (grid.x * BLOCK_SIZE != width) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZE != height) {
        grid.y += 1;
    }

    kernel_calculate_mpr_entry_exit_points <<<grid,block>>>(entry_surf, exit_surf, 
        width, height, mat_mvp_inv, volume_dim, thickness, ray_dir);

    return cudaThreadSynchronize();
}