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

#include "mi_cuda_graphic.h"
#include "arithmetic/mi_cuda_math.h"

__device__ float4 kernel_ray_cast(cudaVolumeInfos* volume_infos, cudaRayCastInfos* ray_cast_infos, float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color) {
    float4 integral_color = input_color;
    float3 sample_pos;
    float3 sample_norm;
    float4 color_ori;
    const float3 dim3_r = make_float3(1.0/volume_infos->dim.x, 1.0/volume_infos->dim.y, 1.0/volume_infos->dim.z);
    float ww,wl,min_gray,gray;
    for (float i = start_step; i < end_step; i+=1.0) {
        sample_pos = ray_start + ray_dir*i;
        sample_norm = sample_pos*dim3_r;

        ww = ray_cast_infos->d_wl_array[0];
        wl = ray_cast_infos->d_wl_array[1];

        min_gray = wl - ww*0.5;

        ///Composite
        gray= tex3D<float>(volume_infos->volume_tex_obj, sample_norm.x,sample_norm.y,sample_norm.z);
        gray = (gray - min_gray)/ww;
        gray = clamp(gray,0.0,1.0);
        color_ori = tex1D<float4>(ray_cast_infos->lut_tex_obj, gray);
        if (color_ori.w > 0.0) {
            integral_color.x += color_ori.x * color_ori.w*(1-integral_color.w);
            integral_color.y += color_ori.y * color_ori.w*(1-integral_color.w);
            integral_color.z += color_ori.z * color_ori.w*(1-integral_color.w);
            integral_color.w += color_ori.w *(1-integral_color.w);
        }

        if (integral_color.w > 0.95) {
            integral_color.w = 1.0;
            break;
        }
    }

    return integral_color;
}

__device__ int kernel_preprocess(float3 entry, float3 exit, float sample_step, float3* ray_start, float3* ray_dir, float* start_step, float* end_step) {
    float3 ray_dir0 = exit - entry;
    float3 ray_dir_norm = normalize(ray_dir0);
    float ray_length = length(ray_dir0);
    if(ray_length < 1e-5) {
        return -1;
    } 

    *ray_start = entry;
    *ray_dir = ray_dir_norm*make_float3(sample_step);
    *start_step = 0;
    *end_step = ray_length/sample_step;

    return 0;
}

__global__ void kernel_ray_cast_main(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height, cudaVolumeInfos volume_infos, cudaRayCastInfos ray_cast_infos, unsigned char* result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width-1 || y > height-1) {
        return;
    }
    uint idx = y*width + x;

    float4 entry = tex2D<float4>(entry_tex, x, y);
    float4 exit  = tex2D<float4>(exit_tex, x, y);

    float3 entry3 = make_float3(entry);
    float3 exit3 = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    /////////////////////////////////////////
    //debug
    //result[idx*4] = exit.x/volume_infos.dim.x*255;
    //result[idx*4+1] = exit.y/volume_infos.dim.y*255;
    //result[idx*4+2] = exit.z/volume_infos.dim.z*255;
    //result[idx*4+3] = 255;

    //return;
    /////////////////////////////////////////

    if(0 != kernel_preprocess(entry3, exit3, ray_cast_infos.sample_step, &ray_start, &ray_dir, &start_step, &end_step)) {
        result[idx*4] = 0;
        result[idx*4+1] = 0;
        result[idx*4+2] = 0;
        result[idx*4] = 0;
        return;
    }

    //__syncthreads();

    float4 input_color = make_float4(0);
    float4 integral_color = kernel_ray_cast(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, input_color );
    
    //__syncthreads();
    clamp(integral_color,0.0,1.0);
    result[idx*4] = integral_color.x*255;
    result[idx*4+1] = integral_color.y*255;
    result[idx*4+2] = integral_color.z*255;
    result[idx*4+3] = 255;
}

__global__ void kernel_ray_cast_main_whole(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height,  cudaVolumeInfos volume_infos, cudaRayCastInfos ray_cast_infos, unsigned char* result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width-1 || y > height-1) {
        return;
    }
    uint idx = y*width + x;

    float4 entry = tex2D<float4>(entry_tex, x, y);
    float4 exit  = tex2D<float4>(exit_tex, x, y);

    float3 entry3 = make_float3(entry);
    float3 exit3 = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    /////////////////////////////////////////
    //debug
    //result[idx*4] = exit.x/volume_infos.dim.x*255;
    //result[idx*4+1] = exit.y/volume_infos.dim.y*255;
    //result[idx*4+2] = exit.z/volume_infos.dim.z*255;
    //result[idx*4+3] = 255;

    //return;
    /////////////////////////////////////////

    float3 ray_dir0 = exit3 - entry3;
    float3 ray_dir_norm = normalize(ray_dir0);
    float ray_length = length(ray_dir0);
    if(ray_length < 1e-5) {
        result[idx*4] = 0;
        result[idx*4+1] = 0;
        result[idx*4+2] = 0;
        result[idx*4+3] = 0;
        return;
    } 

    ray_start = entry3;
    ray_dir = ray_dir_norm*make_float3(ray_cast_infos.sample_step);
    start_step = 0;
    end_step = ray_length/ray_cast_infos.sample_step;

    //__syncthreads();

    float ww = ray_cast_infos.d_wl_array[0];
    float wl = ray_cast_infos.d_wl_array[1];
    float min_gray = wl - ww*0.5;
    float3 dim3_r = make_float3(1.0/volume_infos.dim.x, 1.0/volume_infos.dim.y, 1.0/volume_infos.dim.z);

    float4 integral_color = make_float4(0);
    float3 sample_pos;
    float3 sample_norm;
    float4 color_ori;
    float gray;
    for (float i = start_step; i < end_step; i+=1.0) {
        sample_pos = ray_start + ray_dir*i;
        sample_norm = sample_pos*dim3_r;

        ///Composite
        gray= tex3D<float>(volume_infos.volume_tex_obj, sample_norm.x,sample_norm.y,sample_norm.z);
        gray = (gray - min_gray)/ww;
        gray = clamp(gray,0.0,1.0);
        color_ori = tex1D<float4>(ray_cast_infos.lut_tex_obj, gray);
        if (color_ori.w > 0.0) {
            integral_color.x += color_ori.x * color_ori.w*(1-integral_color.w);
            integral_color.y += color_ori.y * color_ori.w*(1-integral_color.w);
            integral_color.z += color_ori.z * color_ori.w*(1-integral_color.w);
            integral_color.w += color_ori.w *(1-integral_color.w);
        }


        if (integral_color.w > 0.95) {
            integral_color.w = 1.0;
            break;
        }
    }
    
    //__syncthreads();
    clamp(integral_color,0.0,1.0);
    result[idx*4] = integral_color.x*255;
    result[idx*4+1] = integral_color.y*255;
    result[idx*4+2] = integral_color.z*255;
    result[idx*4+3] = 255;
}

//result will be one of color, JEPG buffer.
extern "C"  
int ray_cast(cudaGLTextureReadOnly& entry_tex, cudaGLTextureReadOnly& exit_tex, int width , int height, 
             cudaVolumeInfos volume_info, cudaRayCastInfos ray_cast_info, unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex) {
    //1 launch ray cast kernel
    
    CHECK_CUDA_ERROR;
    map_image(entry_tex);
    map_image(exit_tex);
    CHECK_CUDA_ERROR;

    #define BLOCK_SIZE 16
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    kernel_ray_cast_main<<<grid, block>>>(entry_tex.cuda_tex_obj, exit_tex.cuda_tex_obj, width, height, volume_info, ray_cast_info, d_result);

    //2 JPEG compress(optional)


    //3 Memcpy device result to device GL texture
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    unmap_image(entry_tex);
    unmap_image(exit_tex);
    CHECK_CUDA_ERROR; 

    map_image(canvas_tex);

    write_image(canvas_tex, d_result, width*height * 4);

    unmap_image(canvas_tex);

    CHECK_CUDA_ERROR;

    return 0;
}

extern "C"
int init_data(cudaVolumeInfos& cuda_volume_infos, unsigned short* data, unsigned int* dim) {
    const unsigned int size = dim[0]*dim[1]*dim[2]*sizeof(unsigned short);

    cuda_volume_infos.dim.x = dim[0];
    cuda_volume_infos.dim.y = dim[1];
    cuda_volume_infos.dim.z = dim[2];

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(16,0,0,0,cudaChannelFormatKindUnsigned);
    CHECK_CUDA_ERROR;

    cudaExtent extent;
    extent.width = dim[0];
    extent.height = dim[1];
    extent.depth = dim[2];
    cudaMalloc3DArray(&cuda_volume_infos.d_volume_array, &channel_desc, extent);

    CHECK_CUDA_ERROR;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void *)data,extent.width*sizeof(unsigned short), extent.width, extent.height);
    copyParams.dstArray = cuda_volume_infos.d_volume_array;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    CHECK_CUDA_ERROR;

    //Cuda resource
    struct cudaResourceDesc  res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_volume_infos.d_volume_array;
    
    //Texture parameter (like GL's glTexParameteri)
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc,0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;

    //create texture
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&cuda_volume_infos.volume_tex_obj, &res_desc, &tex_desc, NULL);

    CHECK_CUDA_ERROR;

    return 0;
}


extern "C"
int init_wl_nonmask(cudaRayCastInfos& ray_cast_infos, float* wl_array_norm) {
    CHECK_CUDA_ERROR;
    cudaMalloc(&ray_cast_infos.d_wl_array, sizeof(float)*2);
    cudaMemcpy(ray_cast_infos.d_wl_array, wl_array_norm, sizeof(float)*2, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR;
    return 0;
}

extern "C"
int init_lut_nonmask(cudaRayCastInfos& ray_cast_infos, unsigned char* lut_array, int lut_length) {
    //CUDA array
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&ray_cast_infos.d_lut_array, &channel_desc, lut_length, 1);

    CHECK_CUDA_ERROR;

    //copy data to CUDA array
    cudaMemcpyToArray(ray_cast_infos.d_lut_array, 0, 0, lut_array, lut_length * 4, cudaMemcpyHostToDevice);

    CHECK_CUDA_ERROR;

    //Cuda resource
    struct cudaResourceDesc  res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = ray_cast_infos.d_lut_array;
    
    //Texture parameter (like GL's glTexParameteri)
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc,0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;

    //create texture
    cudaCreateTextureObject(&ray_cast_infos.lut_tex_obj, &res_desc, &tex_desc, NULL);

    CHECK_CUDA_ERROR; 

    ray_cast_infos.lut_length = lut_length;


    return 0;
}

extern "C"
int init_material_nonmask(cudaRayCastInfos& ray_cast_infos, float* material_array) {
    cudaMalloc(&ray_cast_infos.d_material_array, 4*sizeof(float)*3);
    CHECK_CUDA_ERROR; 
    cudaMemcpy(ray_cast_infos.d_material_array, material_array, 4*sizeof(float)*3, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR; 
    return 0;
}