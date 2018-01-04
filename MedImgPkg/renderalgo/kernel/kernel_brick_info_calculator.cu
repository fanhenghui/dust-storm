#include <cuda_runtime.h>
#include "arithmetic/mi_cuda_math.h"
#include "renderalgo/mi_brick_define.h"

using namespace medical_imaging;

__global__ void kernel_cuda_calculate_volume_brick_info(cudaTextureObject_t volume_tex, dim3 volume_dim, float volume_min_scalar,
    int brick_size, dim3 brick_dim, dim3 brick_margin, VolumeBrickInfo* d_data) {

}

__global__ void kernel_cuda_calculate_mask_brick_info(cudaTextureObject_t mask_tex, dim3 mask_dim,
    int brick_size, dim3 brick_dim, dim3 brick_margin,
    dim3 brick_range_min, dim3 brick_range_dim,
    int* d_visible_lables, int visible_label_count,
    MaskBrickInfo* d_data) {

}

extern "C"
cudaError_t cuda_calculate_volume_brick_info(cudaTextureObject_t volume_tex, dim3 volume_dim, float volume_min_scalar,
    int brick_size, dim3 brick_dim, dim3 brick_margin, VolumeBrickInfo* d_data) {
    dim3 block_dim(5, 5, 1);
    dim3 grid_dim(
        brick_dim.x / block_dim.x,
        brick_dim.y / block_dim.y,
        brick_dim.z / block_dim.z);
    if (grid_dim.x * block_dim.x != brick_dim.x) {
        grid_dim.x += 1;
    }
    if (grid_dim.y * block_dim.y != brick_dim.y) {
        grid_dim.y += 1;
    }
    if (grid_dim.z * block_dim.z != brick_dim.z) {
        grid_dim.z += 1;
    }

    kernel_cuda_calculate_volume_brick_info << <grid_dim, block_dim >> > (volume_tex, volume_dim, volume_min_scalar,
        brick_size, brick_dim, brick_margin, d_data);

    return cudaThreadSynchronize();
}

extern "C"
cudaError_t cuda_calculate_mask_brick_info(cudaTextureObject_t mask_tex, dim3 mask_dim,
    int brick_size, dim3 brick_dim, dim3 brick_margin, 
    dim3 brick_range_min, dim3 brick_range_dim,
    int* d_visible_lables, int visible_label_count, 
    MaskBrickInfo* d_data) {
    
    dim3 block_dim(8, 8, 1);
    dim3 grid_dim(
        brick_dim.x / block_dim.x,
        brick_dim.y / block_dim.y,
        brick_dim.z / block_dim.z);
    if (grid_dim.x * block_dim.x != brick_dim.x) {
        grid_dim.x += 1;
    }
    if (grid_dim.y * block_dim.y != brick_dim.y) {
        grid_dim.y += 1;
    }
    if (grid_dim.z * block_dim.z != brick_dim.z) {
        grid_dim.z += 1;
    }

    kernel_cuda_calculate_mask_brick_info << <grid_dim, block_dim >> > (mask_tex, mask_dim, 
        brick_size, brick_dim, brick_margin, 
        brick_range_min, brick_range_dim, 
        d_visible_lables, visible_label_count, d_data);

    return cudaThreadSynchronize();
}