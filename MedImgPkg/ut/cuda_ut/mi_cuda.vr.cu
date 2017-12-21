#include "GL/glew.h"

//CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>  
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <math_functions.h>
#include <vector_functions.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "mi_cuda_vr.h"
#include "arithmetic/mi_cuda_math.h"

//-------------------------------------------//
//Global Parameter Define

#define CHECK_CUDA_ERROR {\
cudaError_t err = cudaGetLastError(); \
if (err != cudaSuccess) {\
    std::cout << "CUDA error: " << err << " in function: " << __FUNCTION__ <<\
    " line: " << __LINE__ << std::endl; \
}}\

//-------------------------------------------//


__device__ void composite() { 

}

__device__ void kernel_ray_cast(float3 ray_dir, float start_step, float end_step, cudaVolumeInfos volume_infos, float4* color_norm) {
    float3 sample_pos;
    for (float i = start_step; i < end_step; i+=1.0) {
        sample_pos = start_step + ray_dir*i;
        //composite();
        if ((*color_norm).w > 0.95) {
            (*color_norm).w = 1.0;
            break;
        }
    }
}

__device__ int kernel_preprocess(float3 entry, float3 exit, float sample_step, float3* ray_start, float3* ray_dir, float* start_step, float* end_step) {
    float3 ray_dir0 = entry - exit;
    float3 ray_dir_norm = normalize(ray_dir0);
    float ray_length = length(ray_dir0);
    if(ray_length < 1e-5) {
        return -1;
    } 

    *ray_start = entry;
    *ray_dir = ray_dir0*make_float3(sample_step);
    *start_step = 0;
    *end_step = ray_length/sample_step;

    return 0;
}

__global__ void kernel_ray_cast_main(cudaGLTexture entry_tex, cudaGLTexture exit_tex, cudaVolumeInfos volume_info, cudaRayCastInfos ray_cast_info, unsigned char* result) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > entry_tex.width-1 || y > entry_tex.height-1) {
        return;
    }
    unsigned int idx = y*entry_tex.width + x;

    float4 entry = tex2D<float4>(entry_tex.cuda_tex_obj, x, y);
    float4 exit = tex2D<float4>(exit_tex.cuda_tex_obj, x, y);

    float3 entry3 = make_float3(entry);
    float3 exit3 = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    if(0 != kernel_preprocess(entry3, exit3, ray_cast_info.sample_step, &ray_start, &ray_dir, &start_step, &end_step)) {
        result[idx*4] = 0;
        result[idx*4+1] = 0;
        result[idx*4+2] = 0;
        result[idx*4] = 0;
        return;
    }
    
    result[idx*4] = entry.x/volume_info.dim.x*255;
    result[idx*4+1] = entry.y/volume_info.dim.y*255;
    result[idx*4+2] = entry.z/volume_info.dim.z*255;
    result[idx*4+3] = 255;
}

//result will be one of color, JEPG buffer.
extern "C"  
int ray_cast(cudaGLTexture entry_tex, cudaGLTexture exit_tex, cudaVolumeInfos volume_info, cudaRayCastInfos ray_cast_info, unsigned char* d_result, unsigned char* h_result) {
    //1 launch ray cast kernel
    int width = entry_tex.width;
    int height = entry_tex.height;

    cudaGraphicsMapResources(1, &entry_tex.cuda_res);
    cudaGraphicsSubResourceGetMappedArray(&entry_tex.cuda_array, entry_tex.cuda_res, 0,0);

    CHECK_CUDA_ERROR; 
    #define BLOCK_SIZE 16
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    kernel_ray_cast_main<<<grid, block>>>(entry_tex, exit_tex, volume_info, ray_cast_info, d_result);

    //2 JPEG compress(optional)


    //3 Memcpy device result to host, and return
    cudaThreadSynchronize();
    cudaGraphicsUnmapResources(1, &entry_tex.cuda_res);

    cudaMemcpy(h_result, d_result, width*height*4, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR; 

    return 0;
}

extern "C"
int bind_gl_texture(cudaGLTexture& gl_cuda_tex) {
    //1 register GL texture to CUDA Graphic resource
    gl_cuda_tex.cuda_res = NULL;
    cudaGraphicsGLRegisterImage(&gl_cuda_tex.cuda_res, gl_cuda_tex.gl_tex_id, gl_cuda_tex.target, cudaGraphicsRegisterFlagsReadOnly);

    CHECK_CUDA_ERROR;

    //2 map the graphic resource to CUDA array
    gl_cuda_tex.cuda_array = NULL;
    cudaGraphicsMapResources(1, &gl_cuda_tex.cuda_res);
    cudaGraphicsSubResourceGetMappedArray(&gl_cuda_tex.cuda_array, gl_cuda_tex.cuda_res, 0,0);
    CHECK_CUDA_ERROR;

    //3 create CUDA texture by CUDA array
    //CUDA resource
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = gl_cuda_tex.cuda_array;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;
    
    cudaCreateTextureObject(&gl_cuda_tex.cuda_tex_obj, &res_desc, &tex_desc, NULL);

    CHECK_CUDA_ERROR;

    cudaGraphicsUnmapResources(1, &gl_cuda_tex.cuda_res);

    CHECK_CUDA_ERROR;

    return 0;
}



