#include "GL/glew.h"

//CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>  
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <math_functions.h>
#include <vector_functions.h>
#include <device_functions.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "mi_cuda_graphic.h"
#include "mi_cuda_vr_common.h"

//this function will slow than operator f3*f3
inline __device__ float3 f3_mul_f3(float3 a, float3 b) {
    return make_float3(__fmul_ru(a.x,b.x), __fmul_ru(a.y, b.y), __fmul_ru(a.z, b.z));
}

//this function will slow than normalize
inline __device__ float3 intri_normalize(float3 a) {
    float invLen = __fdiv_ru(1.0f , __fsqrt_ru(a.x*a.x + a.y*a.y + a.z*a.z));
    return make_float3(a.x*invLen, a.y*invLen , a.z*invLen);
}

//shared memory to fill mask label realted
// visible label (int) : label_level * sizeof(int), label_level could be 1(none-mask) 8 16 32 64 ... 128
// ww wl array (flaot) : label_level * sizeof(float) * 2
// materal parameter : label_level * sizeof(float) * 9

extern __shared__ float s_array[];


inline __device__ float3 cal_gradient(cudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos_norm) {
    float3 x_s = sample_pos_norm + volume_infos->sample_shift;
    float x = tex3D<float>(volume_infos->volume_tex_obj, sample_pos_norm.x + volume_infos->sample_shift.x, sample_pos_norm.y, sample_pos_norm.z) -
              tex3D<float>(volume_infos->volume_tex_obj, sample_pos_norm.x - volume_infos->sample_shift.x, sample_pos_norm.y, sample_pos_norm.z);

    float y = tex3D<float>(volume_infos->volume_tex_obj, sample_pos_norm.x, sample_pos_norm.y + volume_infos->sample_shift.y, sample_pos_norm.z) -
              tex3D<float>(volume_infos->volume_tex_obj, sample_pos_norm.x, sample_pos_norm.y - volume_infos->sample_shift.y, sample_pos_norm.z);

    float z = tex3D<float>(volume_infos->volume_tex_obj, sample_pos_norm.x, sample_pos_norm.y, sample_pos_norm.z + volume_infos->sample_shift.z) -
              tex3D<float>(volume_infos->volume_tex_obj, sample_pos_norm.x, sample_pos_norm.y, sample_pos_norm.z - volume_infos->sample_shift.z);

    return make_float3(x, y, z);
}

//Phong illumination get material from global memory
inline __device__ float4 shade(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 sample_pos, float3 sample_pos_norm, float3 ray_dir, float4 input_color, int label)
{
    float4 tmp;
    float3 color = make_float3(input_color);

    float3 diffuse_color = make_float3(
        ray_cast_infos->d_material_array[label * 9], 
        ray_cast_infos->d_material_array[label * 9 + 1],
        ray_cast_infos->d_material_array[label * 9 + 2]);
    float diffuse_intensity = ray_cast_infos->d_material_array[label * 9 + 3];

    float3 specular_color = make_float3(
        ray_cast_infos->d_material_array[label * 9 + 4],
        ray_cast_infos->d_material_array[label * 9 + 5],
        ray_cast_infos->d_material_array[label * 9 + 6]);

    float specular_intensity = ray_cast_infos->d_material_array[label * 9 + 7];

    float shineness = ray_cast_infos->d_material_array[label * 9 + 8];

    float3 normal = cal_gradient(volume_infos, sample_pos_norm);
    tmp = ray_cast_infos->mat_normal * make_float4(normal);
    normal = normalize(make_float3(tmp));

    tmp = ray_cast_infos->mat_normal * make_float4(-ray_dir);
    float3 view_dir = normalize(make_float3(tmp));

    float3 light_dir = ray_cast_infos->light_position - sample_pos;
    light_dir = normalize(light_dir);
    tmp = ray_cast_infos->mat_normal * make_float4(light_dir);
    light_dir = normalize(make_float3(tmp));

    //ambient
    float3 ambient_part = ray_cast_infos->ambient_color * ray_cast_infos->ambient_intensity * color;
    
    //diffuse
    float ln = dot(light_dir, normal);
    if (ln < 0.0f) {
        normal = -normal;
        ln = -ln;
    }
    float diffuse = max(ln, 0.0f);
    float3 diffuse_part = diffuse * diffuse_color * diffuse_intensity * color;

    //specular(classic phong)
    float3 r = reflect(-light_dir, normal);
    r = normalize(r);
    float specular = max(dot(r, view_dir), 0.0f);

    specular = __powf(specular, shineness);
    float3 specular_part = specular * specular_color * specular_intensity * color;
    
    float3 output_color = ambient_part + diffuse_part + specular_part;

    //silhouettes enhance alpha
    float fn = 1.0f - ln;
    float kss = 1.0f;
    float kse = 0.5f;
    float ksc = 0.0f;
    float alpha = input_color.w*(0.5f + kss * __powf(fn, kse));
    alpha = clamp(alpha, 0.0f, 1.0f);

    output_color = clamp(output_color, 0.0f, 1.0f);
    return make_float4(output_color, alpha);
}


//Phong illumination get material from shared memory
inline __device__ float4 shade_ext(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 sample_pos, float3 sample_pos_norm, float3 ray_dir, float4 input_color, int label)
{
    float4 tmp;
    float3 color = make_float3(input_color);

    const int material_start = ray_cast_infos->mask_level * (1 + 2);
 
    float3 diffuse_color = make_float3(
        s_array[material_start + label * 9],
        s_array[material_start + label * 9 + 1],
        s_array[material_start + label * 9 + 2]);
    float diffuse_intensity = s_array[material_start + label * 9 + 3];

    float3 specular_color = make_float3(
        s_array[material_start + label * 9 + 4],
        s_array[material_start + label * 9 + 5],
        s_array[material_start + label * 9 + 6]);

    float specular_intensity = s_array[material_start + label * 9 + 7];

    float shineness = s_array[material_start + label * 9 + 8];

    float3 normal = cal_gradient(volume_infos, sample_pos_norm);
    tmp = ray_cast_infos->mat_normal * make_float4(normal);
    normal = normalize(make_float3(tmp));

    tmp = ray_cast_infos->mat_normal * make_float4(-ray_dir);
    float3 view_dir = normalize(make_float3(tmp));

    float3 light_dir = ray_cast_infos->light_position - sample_pos;
    light_dir = normalize(light_dir);
    tmp = ray_cast_infos->mat_normal * make_float4(light_dir);
    light_dir = normalize(make_float3(tmp));

    //ambient
    float3 ambient_part = ray_cast_infos->ambient_color * ray_cast_infos->ambient_intensity * color;

    //diffuse
    float ln = dot(light_dir, normal);
    if (ln < 0.0f) {
        normal = -normal;
        ln = -ln;
    }
    float diffuse = max(ln, 0.0f);
    float3 diffuse_part = diffuse * diffuse_color * diffuse_intensity * color;

    //specular(classic phong)
    float3 r = reflect(-light_dir, normal);
    r = normalize(r);
    float specular = max(dot(r, view_dir), 0.0f);

    specular = __powf(specular, shineness);
    float3 specular_part = specular * specular_color * specular_intensity * color;

    float3 output_color = ambient_part + diffuse_part + specular_part;

    //silhouettes enhance alpha
    float fn = 1.0f - ln;
    float kss = 1.0f;
    float kse = 0.5f;
    float ksc = 0.0f;
    float alpha = input_color.w*(0.5f + kss * __powf(fn, kse));
    alpha = clamp(alpha, 0.0f, 1.0f);

    output_color = clamp(output_color, 0.0f, 1.0f);
    return make_float4(output_color, alpha);
}

inline __device__ int access_mask(cudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos) {
    return (int)tex3D<unsigned char>(volume_infos->mask_tex_obj, sample_pos.x, sample_pos.y, sample_pos.z);
}

inline __device__ void composite(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos, 
    float3 sample_pos, float3 ray_dir, float4* __restrict__ integral_color) 
{
    float3 sample_norm = sample_pos*volume_infos->dim_r;
    //sample_norm = f3_mul_f3(sample_pos,dim3_r);
    
    int label = 0;
    float ww = ray_cast_infos->d_wl_array[label * 2];
    float wl = ray_cast_infos->d_wl_array[label * 2 + 1];
    float min_gray = wl - ww*0.5f;

    float gray = tex3D<float>(volume_infos->volume_tex_obj, sample_norm.x, sample_norm.y, sample_norm.z);
    gray = (gray - min_gray) / ww;
    //gray = clamp(gray,0.0f,1.0f);
    float4 color_ori = tex1D<float4>(ray_cast_infos->lut_tex_obj[0], gray);
    float alpha;
    if (color_ori.w > 0.0f) {
        color_ori = shade(volume_infos, ray_cast_infos, sample_pos, sample_norm, ray_dir, color_ori, label);
        alpha = color_ori.w*(1.0f - integral_color->w);
        integral_color->x += color_ori.x * alpha;
        integral_color->y += color_ori.y * alpha;
        integral_color->z += color_ori.z * alpha;
        integral_color->w += color_ori.w *(1 - integral_color->w);
    }
}

inline __device__ void composite_ext(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 sample_pos, float3 ray_dir, float4* __restrict__ integral_color)
{
    float3 sample_norm = sample_pos*volume_infos->dim_r;
    //sample_norm = f3_mul_f3(sample_pos,dim3_r);

    int label = access_mask(volume_infos, sample_pos);
    if (label == 0) {
        return;
    }

    label = 0;
    int wl_start = ray_cast_infos->mask_level;
    float ww = ray_cast_infos->d_wl_array[label * 2];
    float wl = ray_cast_infos->d_wl_array[label * 2 + 1];
    float min_gray = wl - ww*0.5f;

    float gray = tex3D<float>(volume_infos->volume_tex_obj, sample_norm.x, sample_norm.y, sample_norm.z);
    gray = (gray - min_gray) / ww;
    //gray = clamp(gray,0.0f,1.0f);
    float4 color_ori = tex1D<float4>(ray_cast_infos->lut_tex_obj[0], gray);
    float alpha;
    if (color_ori.w > 0.0f) {
        color_ori = shade_ext(volume_infos, ray_cast_infos, sample_pos, sample_norm, ray_dir, color_ori, label);
        alpha = color_ori.w*(1.0f - integral_color->w);
        integral_color->x += color_ori.x * alpha;
        integral_color->y += color_ori.y * alpha;
        integral_color->z += color_ori.z * alpha;
        integral_color->w += color_ori.w *(1 - integral_color->w);
    }
}


////Composite inline ray cast(FPS is the same with use inline composite sub function)
//__device__ float4 kernel_ray_cast(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos, float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color) {
//    float4 integral_color = input_color;
//    float3 sample_pos;
//    float3 sample_norm;
//    float4 color_ori;
//    float ww,wl,min_gray,gray,alpha;
//
//    for (float i = start_step; i < end_step; i+=1.0f) {
//        sample_pos = ray_start + ray_dir*i;
//        sample_norm = sample_pos*volume_infos->dim_r;
//        //sample_norm = f3_mul_f3(sample_pos,dim3_r);
//        
//
//        ww = ray_cast_infos->d_wl_array[0];
//        wl = ray_cast_infos->d_wl_array[1];
//
//        min_gray = wl - ww*0.5f;
//
//        ///Composite
//        gray= tex3D<float>(volume_infos->volume_tex_obj, sample_norm.x,sample_norm.y,sample_norm.z);
//        gray = (gray - min_gray)/ww;
//        //gray = clamp(gray,0.0f,1.0f);
//        color_ori = tex1D<float4>(ray_cast_infos->lut_tex_obj, gray);
//        if (color_ori.w > 0.0f) {
//            alpha = color_ori.w*(1.0f - integral_color.w);
//            integral_color.x += color_ori.x * alpha;
//            integral_color.y += color_ori.y * alpha;
//            integral_color.z += color_ori.z * alpha;
//            integral_color.w += color_ori.w *(1-integral_color.w);
//        }
//
//        if (integral_color.w > 0.95f) {
//            integral_color.w = 1.0f;
//            break;
//        }
//    }
//
//    return integral_color;
//}

__device__ float4 kernel_ray_cast(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos, float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color) {
    float4 integral_color = input_color;
    float3 sample_pos;

    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        composite(volume_infos, ray_cast_infos, sample_pos, ray_dir, &integral_color);
        if (integral_color.w > 0.95f) {
            integral_color.w = 1.0f;
            break;
        }
    }

    return integral_color;
}

__device__ float4 kernel_ray_cast_ext(cudaVolumeInfos* __restrict__ volume_infos, cudaRayCastInfos* __restrict__ ray_cast_infos, float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color) {
    float4 integral_color = input_color;
    float3 sample_pos;

    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        composite_ext(volume_infos, ray_cast_infos, sample_pos, ray_dir, &integral_color);
        if (integral_color.w > 0.95f) {
            integral_color.w = 1.0f;
            break;
        }
    }

    return integral_color;
}

__device__ int kernel_preprocess(float3 entry, float3 exit, float sample_step, float3* __restrict__ ray_start, float3* __restrict__ ray_dir, float* __restrict__ start_step, float* __restrict__ end_step) {
    float3 ray_dir0 = exit - entry;
    float3 ray_dir_norm = normalize(ray_dir0);
    float ray_length = length(ray_dir0);
    if(ray_length < 1e-5f) {
        return -1;
    } 

    *ray_start = entry;
    *ray_dir = ray_dir_norm*make_float3(sample_step);
    *start_step = 0.0f;
    *end_step = ray_length/sample_step;

    return 0;
}

__global__ void kernel_ray_cast_main(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height, cudaVolumeInfos volume_infos, cudaRayCastInfos ray_cast_infos, unsigned char* __restrict__ result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint idx = y*width + x;

    if (x > width-1 || y > height-1) {
        return;
    }

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

    float4 input_color = make_float4(0.0f);
    float4 integral_color = kernel_ray_cast(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, input_color );
    
    //__syncthreads();
    clamp(integral_color,0.0f,1.0f);
    result[idx * 4] = integral_color.x * 255;
    result[idx * 4 + 1] = integral_color.y * 255;
    result[idx * 4 + 2] = integral_color.z * 255;
    result[idx * 4 + 3] = 255;
}

__global__ void kernel_ray_cast_main_ext(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height, cudaVolumeInfos volume_infos, cudaRayCastInfos ray_cast_infos, unsigned char* __restrict__ result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    //fill 
    uint local_thread = threadIdx.y * blockDim.x + threadIdx.x;
    if (local_thread < ray_cast_infos.mask_level) {
        int wl_start = ray_cast_infos.mask_level;
        s_array[wl_start + local_thread * 2] = ray_cast_infos.d_wl_array[local_thread * 2];
        s_array[wl_start + local_thread * 2 + 1] = ray_cast_infos.d_wl_array[local_thread * 2 + 1];
        int material_start = wl_start + ray_cast_infos.mask_level*2;
        for (int i = 0; i < 9; i++)
        {
            s_array[material_start + local_thread * 9 + i] = ray_cast_infos.d_material_array[local_thread *9 + i];
        }
    }
    __syncthreads();


    if (x > width - 1 || y > height - 1) {
        return;
    }
    uint idx = y*width + x;

    float4 entry = tex2D<float4>(entry_tex, x, y);
    float4 exit = tex2D<float4>(exit_tex, x, y);

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

    if (0 != kernel_preprocess(entry3, exit3, ray_cast_infos.sample_step, &ray_start, &ray_dir, &start_step, &end_step)) {
        result[idx * 4] = 0;
        result[idx * 4 + 1] = 0;
        result[idx * 4 + 2] = 0;
        result[idx * 4] = 0;
        return;
    }

    //__syncthreads();

    float4 input_color = make_float4(0.0f);
    float4 integral_color = kernel_ray_cast_ext(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, input_color);

    //__syncthreads();
    clamp(integral_color, 0.0f, 1.0f);
    result[idx * 4] = integral_color.x * 255;
    result[idx * 4 + 1] = integral_color.y * 255;
    result[idx * 4 + 2] = integral_color.z * 255;
    result[idx * 4 + 3] = 255;
    
}


//This is write for compare with sub function (FPS is the same)
//__global__ void kernel_ray_cast_main_whole(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height,  cudaVolumeInfos volume_infos, cudaRayCastInfos ray_cast_infos, unsigned char* result) {
//    uint x = blockIdx.x * blockDim.x + threadIdx.x;
//    uint y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x > width-1 || y > height-1) {
//        return;
//    }
//    uint idx = y*width + x;
//
//    float4 entry = tex2D<float4>(entry_tex, x, y);
//    float4 exit  = tex2D<float4>(exit_tex, x, y);
//
//    float3 entry3 = make_float3(entry);
//    float3 exit3 = make_float3(exit);
//
//    float3 ray_start, ray_dir;
//    float start_step, end_step;
//
//    /////////////////////////////////////////
//    //debug
//    //result[idx*4] = exit.x/volume_infos.dim.x*255;
//    //result[idx*4+1] = exit.y/volume_infos.dim.y*255;
//    //result[idx*4+2] = exit.z/volume_infos.dim.z*255;
//    //result[idx*4+3] = 255;
//
//    //return;
//    /////////////////////////////////////////
//
//    float3 ray_dir0 = exit3 - entry3;
//    float3 ray_dir_norm = normalize(ray_dir0);
//    float ray_length = length(ray_dir0);
//    if(ray_length < 1e-5f) {
//        result[idx*4] = 0;
//        result[idx*4+1] = 0;
//        result[idx*4+2] = 0;
//        result[idx*4+3] = 0;
//        return;
//    } 
//
//    ray_start = entry3;
//    ray_dir = ray_dir_norm*make_float3(ray_cast_infos.sample_step);
//    start_step = 0.0f;
//    end_step = ray_length/ray_cast_infos.sample_step;
//
//    //__syncthreads();
//
//    float ww = ray_cast_infos.d_wl_array[0];
//    float wl = ray_cast_infos.d_wl_array[1];
//    float min_gray = wl - ww*0.5f;
//    float3 dim3_r = make_float3(1.0f/volume_infos.dim.x, 1.0f/volume_infos.dim.y, 1.0f/volume_infos.dim.z);
//
//    float4 integral_color = make_float4(0);
//    float3 sample_pos;
//    float3 sample_norm;
//    float4 color_ori;
//    float gray;
//    for (float i = start_step; i < end_step; i+=1.0f) {
//        sample_pos = ray_start + ray_dir*i;
//        sample_norm = sample_pos*dim3_r;
//
//        ///Composite
//        gray= tex3D<float>(volume_infos.volume_tex_obj, sample_norm.x,sample_norm.y,sample_norm.z);
//        gray = (gray - min_gray)/ww;
//        gray = clamp(gray,0.0f,1.0f);
//        color_ori = tex1D<float4>(ray_cast_infos.lut_tex_obj, gray);
//        if (color_ori.w > 0.0f) {
//            integral_color.x += color_ori.x * color_ori.w*(1-integral_color.w);
//            integral_color.y += color_ori.y * color_ori.w*(1-integral_color.w);
//            integral_color.z += color_ori.z * color_ori.w*(1-integral_color.w);
//            integral_color.w += color_ori.w *(1-integral_color.w);
//        }
//
//
//        if (integral_color.w > 0.95f) {
//            integral_color.w = 1.0f;
//            break;
//        }
//    }
//    
//    //__syncthreads();
//    clamp(integral_color,0.0f,1.0f);
//    result[idx*4] = integral_color.x*255;
//    result[idx*4+1] = integral_color.y*255;
//    result[idx*4+2] = integral_color.z*255;
//    result[idx*4+3] = 255;
//}

//result will be one of color, JEPG buffer.
extern "C"  
int ray_cast(cudaGLTextureReadOnly& entry_tex, cudaGLTextureReadOnly& exit_tex, int width , int height, 
             cudaVolumeInfos volume_info, cudaRayCastInfos ray_cast_info, unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex, bool d_cpy) {
    //1 launch ray cast kernel
    
    CHECK_CUDA_ERROR;
    map_image(entry_tex);
    map_image(exit_tex);
    CHECK_CUDA_ERROR;

    #define BLOCK_SIZE 16
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    //kernel_ray_cast_main<<<grid, block>>>(entry_tex.cuda_tex_obj, exit_tex.cuda_tex_obj, width, height, volume_info, ray_cast_info, d_result);

    kernel_ray_cast_main_ext << <grid, block , ray_cast_info.mask_level*(1+2+9)*sizeof(float)>> >(entry_tex.cuda_tex_obj, exit_tex.cuda_tex_obj, width, height, volume_info, ray_cast_info, d_result);

    //2 JPEG compress(optional)


    //3 Memcpy device result to device GL texture
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    unmap_image(entry_tex);
    unmap_image(exit_tex);
    CHECK_CUDA_ERROR; 

    if (d_cpy) {
        map_image(canvas_tex);

        write_image(canvas_tex, d_result, width*height * 4);

        unmap_image(canvas_tex);

        CHECK_CUDA_ERROR;
    }

    return 0;
}

extern "C" 
int init_mask(cudaVolumeInfos& cuda_volume_infos, unsigned char* data) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    CHECK_CUDA_ERROR;

    cudaExtent extent;
    extent.width = cuda_volume_infos.dim.x;
    extent.height = cuda_volume_infos.dim.y;
    extent.depth = cuda_volume_infos.dim.z;
    cudaMalloc3DArray(&cuda_volume_infos.d_mask_array, &channel_desc, extent);

    CHECK_CUDA_ERROR;

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void *)data, extent.width * sizeof(unsigned char), extent.width, extent.height);
    copyParams.dstArray = cuda_volume_infos.d_mask_array;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    CHECK_CUDA_ERROR;

    //Cuda resource
    struct cudaResourceDesc  res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_volume_infos.d_mask_array;

    //Texture parameter (like GL's glTexParameteri)
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    //create texture
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&cuda_volume_infos.mask_tex_obj, &res_desc, &tex_desc, NULL);

    CHECK_CUDA_ERROR;

    return 0;
}

extern "C"
int init_data(cudaVolumeInfos& cuda_volume_infos, unsigned short* data) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(16,0,0,0,cudaChannelFormatKindUnsigned);
    CHECK_CUDA_ERROR;

    cudaExtent extent;
    extent.width = cuda_volume_infos.dim.x;
    extent.height = cuda_volume_infos.dim.y;
    extent.depth = cuda_volume_infos.dim.z;
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
    cudaCreateTextureObject(&ray_cast_infos.lut_tex_obj[0], &res_desc, &tex_desc, NULL);

    ray_cast_infos.lut_tex_obj[1] = ray_cast_infos.lut_tex_obj[0];
    CHECK_CUDA_ERROR; 

    ray_cast_infos.lut_length = lut_length;


    return 0;
}

extern "C"
int init_material(cudaRayCastInfos& ray_cast_infos, float* material_array, int mask_level) {
    cudaMalloc(&ray_cast_infos.d_material_array, mask_level * 9 * sizeof(float));
    CHECK_CUDA_ERROR;
    cudaMemcpy(ray_cast_infos.d_material_array, material_array, mask_level * 9 * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR;
    return 0;
}

extern "C"
int init_material_nonmask(cudaRayCastInfos& ray_cast_infos, float* material_array) {
    return init_material(ray_cast_infos, material_array, 1);
}



__global__ void kernel_rgba8_to_rgb8(uint width, uint height, unsigned char* __restrict__ d_rgba, unsigned char* __restrict__ d_rgb) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height) {
        return;
    }
    uint idx = y*width + x;

    d_rgb[idx * 3] = d_rgba[idx * 4];
    d_rgb[idx * 3 + 1] = d_rgba[idx * 4 + 1];
    d_rgb[idx * 3 + 2] = d_rgba[idx * 4 + 2];
}

extern "C"
int rgba8_to_rgb8(int width , int height ,  unsigned char*  d_rgba , unsigned char*  d_rgb) {
    CHECK_CUDA_ERROR;

    const int BLOCKDIM = 16;
    dim3 block_dim(BLOCKDIM, BLOCKDIM);
    dim3 grid_dim(width/BLOCKDIM, height/BLOCKDIM);
    kernel_rgba8_to_rgb8<<<grid_dim , block_dim>>>(width, height, d_rgba, d_rgb);


    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    return 0;
}

__global__ void kernel_rgba8_to_rgb8_mirror(uint width, uint height, unsigned char* __restrict__ d_rgba, unsigned char* __restrict__ d_rgb) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height) {
        return;
    }
    uint idx = y*width + x;
    uint idx2 = (height-1-y)*width + x;

    d_rgb[idx * 3] = d_rgba[idx2 * 4];
    d_rgb[idx * 3 + 1] = d_rgba[idx2 * 4 + 1];
    d_rgb[idx * 3 + 2] = d_rgba[idx2 * 4 + 2];
}

extern "C"
int rgba8_to_rgb8_mirror(int width, int height, unsigned char*  d_rgba, unsigned char*  d_rgb) {
    CHECK_CUDA_ERROR;

    const int BLOCKDIM = 16;
    dim3 block_dim(BLOCKDIM, BLOCKDIM);
    dim3 grid_dim(width / BLOCKDIM, height / BLOCKDIM);
    kernel_rgba8_to_rgb8_mirror << <grid_dim, block_dim >> >(width, height, d_rgba, d_rgb);


    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;

    return 0;
}
