#include <cuda_runtime.h>
#include "arithmetic/mi_cuda_math.h"

__global__ void kernel_surface_2d_rgba8_flip_vertical_to_global_memory_rgb8(cudaSurfaceObject_t sur_rgba8, int width, int height, unsigned char* d_rgb_8) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 rgba;
        surf2Dread(&rgba, sur_rgba8, x * 4, y);
        unsigned int idx = (height - y) * width + x;
        d_rgb_8[idx * 3] = rgba.x;
        d_rgb_8[idx * 3 + 1] = rgba.y;
        d_rgb_8[idx * 3 + 2] = rgba.z;
    }
}

extern "C"
cudaError_t surface_2d_rgba8_flip_vertical_to_global_memory_rgb8(cudaSurfaceObject_t sur_rgba8, int width, int height, unsigned char* d_rgb_8) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    if (grid.x * BLOCK_SIZE != width) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZE != height) {
        grid.y += 1;
    }

    kernel_surface_2d_rgba8_flip_vertical_to_global_memory_rgb8 << <grid, block>> >(sur_rgba8, width, height, d_rgb_8);

    return cudaThreadSynchronize();
}