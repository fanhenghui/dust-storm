#include <cuda_runtime.h>
#include "arithmetic/mi_cuda_math.h"
#include "renderalgo/mi_brick_define.h"

using namespace medical_imaging;


__device__ void kernel_statistic_volume_info_in_cube(int3 begin, int3 end, cudaTextureObject_t volume_tex, float *min0, float* max0) {
    unsigned short min00 = 65535;
    unsigned short max00 = 0;
    unsigned short v = 0;
    for (int z = begin.z; z < end.z; ++z) {
        for (int y = begin.y; y < end.y; ++y) {
            for (int x = begin.x; x < end.x; ++x) {
                v = tex3D<unsigned short>(volume_tex, x, y, z);
                min00 = min(min00, v);
                max00 = max(max00, v);
            }
        }
    }
    *min0 = float(min00);
    *max0 = float(max00);
}

__global__ void kernel_cuda_calculate_volume_brick_info(cudaTextureObject_t volume_tex, dim3 volume_dim,
    int brick_size, dim3 brick_dim, int brick_margin, VolumeBrickInfo* d_data) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > brick_dim.x - 1 || y > brick_dim.y - 1 || z > brick_dim.z - 1) {
        return;
    }

    unsigned int idx = z * brick_dim.x * brick_dim.y + y * brick_dim.x + x;
    int3 begin = make_int3(x*brick_size , y*brick_size, z*brick_size);
    int3 end = begin + make_int3(brick_size);
    int3 margin = make_int3(brick_margin);

    begin -= margin;
    end += margin;

    begin = max(begin, make_int3(0));
    begin = min(begin, make_int3(volume_dim));
    
    end = max(end, make_int3(0));
    end = min(end, make_int3(volume_dim));

    float max0 = 0.0f;
    float min0 = 0.0f;
    kernel_statistic_volume_info_in_cube(begin, end, volume_tex, &min0, &max0);
    d_data[idx].min = min0;
    d_data[idx].max = max0;
}

__shared__ unsigned char _s_visible_labels[256];
__device__ void kernel_statistic_mask_info_in_cube(int3 begin, int3 end, cudaTextureObject_t mask_tex, int* label_code) {
    unsigned char label_max = 0;
    unsigned char label_min = 255;
    unsigned char label = 0;
    int all_air = 1;
    for (int z = begin.z; z < end.z; ++z) {
        for (int y = begin.y; y < end.y; ++y) {
            for (int x = begin.x; x < end.x; ++x) {
                label = tex3D<unsigned char>(mask_tex, x, y, z);
                if (label == 0) {
                    continue;
                }
                if (_s_visible_labels[label] != 0) {
                    label_min = min(label_min, label);
                    label_max = max(label_max, label);
                    all_air ^= all_air;
                }
                
                
            }
        }
    }

    if (1 == all_air) {
        *label_code = 0;
    } else if (label_min == label_max) {
        *label_code = (int)label_min;
    } else {
        *label_code = 255;
    }
}

__global__ void kernel_cuda_calculate_mask_brick_info(cudaTextureObject_t mask_tex, dim3 mask_dim,
    int brick_size, dim3 brick_dim, int brick_margin,
    dim3 brick_range_min, dim3 brick_range_dim,
    unsigned char* d_visible_label_bucket, int visible_label_bucket_count,
    MaskBrickInfo* d_data) {
    //-------------------------------------------------------------//
    //set shared memory
    if (threadIdx.y * blockDim.x * blockDim.y + threadIdx.x == 0) {
        //reset shared data to 0
        int* shared_map = (int*)_s_visible_labels;
        for (int i = 0; i < 64; ++i) {
            shared_map[i] ^= shared_map[i];
        }
        //write global to shared visible labels
        for (int i = 0; i < visible_label_bucket_count; ++i) {
            _s_visible_labels[i] = d_visible_label_bucket[i];
        }
    }

    __syncthreads();
    //-------------------------------------------------------------//

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > brick_range_dim.x - 1 || y > brick_range_dim.y - 1 || z > brick_range_dim.z - 1) {
        return;
    }

    x += brick_range_min.x;
    y += brick_range_min.y;
    z += brick_range_min.z;
    unsigned int idx = z * brick_dim.x * brick_dim.y + y * brick_dim.x + x;
    
    int3 begin = make_int3(x*brick_size, y*brick_size, z*brick_size);
    int3 end = begin + make_int3(brick_size);
    int3 margin = make_int3(brick_margin);

    begin -= margin;
    end += margin;

    begin = max(begin, make_int3(0));
    begin = min(begin, make_int3(mask_dim));

    end = max(end, make_int3(0));
    end = min(end, make_int3(mask_dim));

    int label_code = 0;
    kernel_statistic_mask_info_in_cube(begin, end, mask_tex, &label_code);
    d_data[idx].label = label_code;
}

extern "C"
cudaError_t cuda_calculate_volume_brick_info(cudaTextureObject_t volume_tex, dim3 volume_dim,
    int brick_size, dim3 brick_dim, int brick_margin, VolumeBrickInfo* d_data) {
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
    
    kernel_cuda_calculate_volume_brick_info << <grid_dim, block_dim >> > (volume_tex, volume_dim,
        brick_size, brick_dim, brick_margin, d_data);

    return cudaThreadSynchronize();
}

extern "C"
cudaError_t cuda_calculate_mask_brick_info(cudaTextureObject_t mask_tex, dim3 mask_dim,
    int brick_size, dim3 brick_dim, int brick_margin, 
    dim3 brick_range_min, dim3 brick_range_dim,
    unsigned char* d_visible_label_bucket, int visible_label_bucket_count,
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
        d_visible_label_bucket, visible_label_bucket_count, d_data);

    return cudaThreadSynchronize();
}