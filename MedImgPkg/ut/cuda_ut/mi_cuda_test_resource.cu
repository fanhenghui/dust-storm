#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda.h>
#include "arithmetic/mi_cuda_math.h"
#include  "mi_cuda_vr_common.h"

__global__ void kernel_draw_slice(int i, cudaTextureObject_t tex, dim3 dim, int width, int height, unsigned char* d_rgba) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int idx = y*width + x;

    if (x == 512 || y == 512) {
        d_rgba[idx * 4 + 0] = 255;
        d_rgba[idx * 4 + 1] = 0;
        d_rgba[idx * 4 + 2] = 0;
        d_rgba[idx * 4 + 3] = 0;
        return;
    }

    if (x == dim.x || y ==dim.y) {
        d_rgba[idx * 4 + 0] = 255;
        d_rgba[idx * 4 + 1] = 255;
        d_rgba[idx * 4 + 2] = 0;
        d_rgba[idx * 4 + 3] = 0;
        return;
    }

    
    if (x > dim.x - 1 || y > dim.y - 1) {
        d_rgba[idx * 4 + 0] = 0;
        d_rgba[idx * 4 + 1] = 0;
        d_rgba[idx * 4 + 2] = 0;
        d_rgba[idx * 4 + 3] = 0;
        return;
    }
    
    //read original type
    /*unsigned short v = tex3D<unsigned short>(tex, x, y, i);
    float v_f = v / 512.0f;
    v_f = clamp(v_f,0.0f,1.0f);
    unsigned char v_uc = (unsigned char)(v_f*255.0f);*/

    //read normalized
    float v = tex3D<float>(tex, x, y, i);
    float v_f = v * 65535.0f / 512.0f;
    v_f = clamp(v_f, 0.0f, 1.0f);
    unsigned char v_uc = (unsigned char)(v_f*255.0f);

    d_rgba[idx * 4 + 0] = v_uc;
    d_rgba[idx * 4 + 1] = v_uc;
    d_rgba[idx * 4 + 2] = v_uc;
    d_rgba[idx * 4 + 3] = 255;
}

extern "C"
void draw_slice(int i, cudaTextureObject_t tex, dim3 dim, int width, int height, unsigned char* d_rgba) {
    dim3 block_size(16,16);
    dim3 grid_size(width/16, height/16);
    CHECK_CUDA_ERROR;

    kernel_draw_slice<<<grid_size, block_size>>>(i, tex, dim, width, height, d_rgba);

    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;
}