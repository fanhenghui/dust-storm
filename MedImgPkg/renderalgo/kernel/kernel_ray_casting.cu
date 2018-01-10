#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_functions.h>

#include "renderalgo/mi_ray_caster_define.h"


//this function will slow than operator f3*f3
inline __device__ float3 f3_mul_f3(float3 a, float3 b) {
    return make_float3(__fmul_ru(a.x, b.x), __fmul_ru(a.y, b.y), __fmul_ru(a.z, b.z));
}

//this function will slow than normalize
inline __device__ float3 intri_normalize(float3 a) {
    float invLen = __fdiv_ru(1.0f, __fsqrt_ru(a.x*a.x + a.y*a.y + a.z*a.z));
    return make_float3(a.x*invLen, a.y*invLen, a.z*invLen);
}

//---------------------------------------------------------//
//shared memory contains follows:
// 1. visible label (int) : label_level * sizeof(int), label_level could be 1(none-mask) 8 16 32 64 ... 128
// 2. ww wl array (flaot) : label_level * sizeof(float) * 2
// 3. color/opacity texture array (tex1D): label_level * sizeof(unsigned long long)
// 4. materal parameter : label_level * sizeof(float) * 9
// sum: label_level * [4*1 + 4*2 + 8*1 + 4*9] = label_level * 56, max : 14KB < shared limits(40KB)
//---------------------------------------------------------//
extern __shared__ float s_array[];

inline __device__ __host__ int get_s_array_size(int label_level) {
    return 56 * label_level;
}

inline __device__ int* get_s_visible_label_array(int label_level) {
    return (int*)(s_array);
}

inline __device__ float* get_s_wl_array(int label_level) {
    return (float*)((char*)(s_array) + 4 * label_level);
}

inline __device__ cudaTextureObject_t* get_s_color_opacity_texture_array(int label_level) {
    return (cudaTextureObject_t*)((char*)(s_array) + 12 * label_level);
}

inline __device__ float* get_s_material_array(int label_level) {
    return (float*)((char*)(s_array) + 20 * label_level);
}

inline __device__ float3 cal_gradient(CudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos_norm) {
    float3 x_s = sample_pos_norm + volume_infos->sample_shift;
    float x = tex3D<float>(volume_infos->volume_tex, sample_pos_norm.x + volume_infos->sample_shift.x, sample_pos_norm.y, sample_pos_norm.z) -
        tex3D<float>(volume_infos->volume_tex, sample_pos_norm.x - volume_infos->sample_shift.x, sample_pos_norm.y, sample_pos_norm.z);

    float y = tex3D<float>(volume_infos->volume_tex, sample_pos_norm.x, sample_pos_norm.y + volume_infos->sample_shift.y, sample_pos_norm.z) -
        tex3D<float>(volume_infos->volume_tex, sample_pos_norm.x, sample_pos_norm.y - volume_infos->sample_shift.y, sample_pos_norm.z);

    float z = tex3D<float>(volume_infos->volume_tex, sample_pos_norm.x, sample_pos_norm.y, sample_pos_norm.z + volume_infos->sample_shift.z) -
        tex3D<float>(volume_infos->volume_tex, sample_pos_norm.x, sample_pos_norm.y, sample_pos_norm.z - volume_infos->sample_shift.z);

    return make_float3(x, y, z);
}

//Phong illumination get material from shared memory
inline __device__ float4 shade_phong(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 sample_pos, float3 sample_pos_norm, float3 ray_dir, float4 input_color, int label) {
    float4 tmp;
    float3 color = make_float3(input_color);

    float * s_material_array = get_s_material_array(ray_cast_infos->label_level);

    float3 diffuse_color = make_float3(
        s_material_array[label * 9],
        s_material_array[label * 9 + 1],
        s_material_array[label * 9 + 2]);
    float diffuse_intensity = s_material_array[label * 9 + 3];

    float3 specular_color = make_float3(
        s_material_array[label * 9 + 4],
        s_material_array[label * 9 + 5],
        s_material_array[label * 9 + 6]);

    float specular_intensity = s_material_array[label * 9 + 7];

    float shineness = s_material_array[label * 9 + 8];

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

inline __device__ int access_mask_nearest(CudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos) {
    return (int)tex3D<unsigned char>(volume_infos->mask_tex, sample_pos.x, sample_pos.y, sample_pos.z);
}

inline __device__ int access_mask_linear(CudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos) {
    return (int)tex3D<unsigned char>(volume_infos->mask_tex, sample_pos.x, sample_pos.y, sample_pos.z);
}

inline __device__ void composite_dvr(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 sample_pos, float3 ray_dir, float4* __restrict__ integral_color) {
    float3 sample_norm = sample_pos*volume_infos->dim_r;
    //sample_norm = f3_mul_f3(sample_pos,dim3_r);

    int label = 0;
    if (1 == ray_cast_infos->mask_mode) {
        label = access_mask_nearest(volume_infos, sample_pos);
    } else if (0 == ray_cast_infos->mask_mode) {
        label = 0;
    } else {
        label = access_mask_linear(volume_infos, sample_pos);
    }

    if (label == 0) {
        return;
    }

    label = 0;
    float* s_wl_array = get_s_wl_array(ray_cast_infos->label_level);
    float ww = s_wl_array[label * 2];
    float wl = s_wl_array[label * 2 + 1];
    float min_gray = wl - ww*0.5f;

    float gray = tex3D<float>(volume_infos->volume_tex, sample_norm.x, sample_norm.y, sample_norm.z);
    gray = (gray - min_gray) / ww;
    //gray = clamp(gray,0.0f,1.0f);
    cudaTextureObject_t* color_opacity_texture_array = get_s_color_opacity_texture_array(ray_cast_infos->label_level);
    float4 color_ori = tex1D<float4>(color_opacity_texture_array[label], gray);
    float alpha;
    if (color_ori.w > 0.0f) {
        if (1 == ray_cast_infos->shading_mode) {
            color_ori = shade_phong(volume_infos, ray_cast_infos, sample_pos, sample_norm, ray_dir, color_ori, label);
        }
        
        alpha = color_ori.w*(1.0f - integral_color->w);
        integral_color->x += color_ori.x * alpha;
        integral_color->y += color_ori.y * alpha;
        integral_color->z += color_ori.z * alpha;
        integral_color->w += color_ori.w *(1 - integral_color->w);
    }
}

__device__ float4 kernel_ray_cast(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos, 
    float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color) {
    float4 integral_color = input_color;
    float3 sample_pos;

    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        composite_dvr(volume_infos, ray_cast_infos, sample_pos, ray_dir, &integral_color);
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
    if (ray_length < 1e-5f) {
        return -1;
    }

    *ray_start = entry;
    *ray_dir = ray_dir_norm*make_float3(sample_step);
    *start_step = 0.0f;
    *end_step = ray_length / sample_step;

    return 0;
}

__device__ void fill_shared_array(int thread_idx, void* d_mapped_array, int size, int memcpy_step) {
    int begin = thread_idx * memcpy_step;
    char* dst = (char*)s_array;
    char* src = (char*)d_mapped_array;
    if (begin < size) {
        int end = min((thread_idx+1)*memcpy_step, size);
        for (int i = begin; i < end; ++i) {
            dst[i] = src[i];
        }
    }
}

__global__ void kernel_ray_cast_main_texture(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height,
    CudaVolumeInfos volume_infos, CudaRayCastInfos ray_cast_infos, cudaSurfaceObject_t canvas, int memcpy_step) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    //fill shared array 
    uint local_thread = threadIdx.y * blockDim.x + threadIdx.x;
    //fill_shared_array(local_thread, ray_cast_infos.d_shared_mapped_memory, get_s_array_size(ray_cast_infos.label_level), memcpy_step);
    //__syncthreads();


    if (x > width - 1 || y > height - 1) {
        return;
    }
    uint idx = y*width + x;

    float4 entry = tex2D<float4>(entry_tex, x, y);
    float4 exit  = tex2D<float4>(exit_tex, x, y);

    float3 entry3 = make_float3(entry);
    float3 exit3  = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    /////////////////////////////////////////
    //debug
    float4 debug_color = entry;
    uchar4 rgba_entry_exit = make_uchar4(debug_color.x / volume_infos.dim.x * 255, debug_color.y / volume_infos.dim.y * 255, debug_color.z / volume_infos.dim.z * 255, 255);
    surf2Dwrite(rgba_entry_exit, canvas, x * 4, y);
    return;
    /////////////////////////////////////////

    if (0 != kernel_preprocess(entry3, exit3, ray_cast_infos.sample_step, &ray_start, &ray_dir, &start_step, &end_step)) {
        uchar4 rgba = make_uchar4(0,0,0,0);
        surf2Dwrite(rgba, canvas, x * 4, y);
        return;
    }

    //__syncthreads();

    float4 input_color = make_float4(0.0f);
    float4 integral_color = kernel_ray_cast(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, input_color);

    //__syncthreads();
    clamp(integral_color, 0.0f, 1.0f);

    uchar4 rgba = make_uchar4(integral_color.x * 255, integral_color.y * 255 , integral_color.z * 255, 255);
    surf2Dwrite(rgba, canvas, x*4 , y);
}

__global__ void kernel_ray_cast_main_surface(cudaSurfaceObject_t entry_surf, cudaSurfaceObject_t exit_surf, int width, int height,
    CudaVolumeInfos volume_infos, CudaRayCastInfos ray_cast_infos, cudaSurfaceObject_t canvas, int memcpy_step) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    //fill shared array
    uint local_thread = threadIdx.y * blockDim.x + threadIdx.x;
    fill_shared_array(local_thread, ray_cast_infos.d_shared_mapped_memory, get_s_array_size(ray_cast_infos.label_level), memcpy_step);
    __syncthreads();


    if (x > width - 1 || y > height - 1) {
        return;
    }
    uint idx = y*width + x;

    float4 entry, exit;
    surf2Dread(&entry, entry_surf, x * 4, y);
    surf2Dread(&exit, exit_surf, x * 4, y);

    float3 entry3 = make_float3(entry);
    float3 exit3 = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    /////////////////////////////////////////
    //debug
    //uchar4 rgba_entry_exit(exit.x / volume_infos.dim.x * 255, exit.y / volume_infos.dim.y * 255, exit.z / volume_infos.dim.z * 255, 255);
    //surf2DWrite(rgba_entry_exit, canvas, x * 4, y);
    //return;
    /////////////////////////////////////////

    if (0 != kernel_preprocess(entry3, exit3, ray_cast_infos.sample_step, &ray_start, &ray_dir, &start_step, &end_step)) {
        uchar4 rgba = make_uchar4(0, 0, 0, 0);
        surf2Dwrite(rgba, canvas, x * 4, y);
        return;
    }

    //__syncthreads();

    float4 input_color = make_float4(0.0f);
    float4 integral_color = kernel_ray_cast(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, input_color);

    //__syncthreads();
    clamp(integral_color, 0.0f, 1.0f);

    uchar4 rgba = make_uchar4(integral_color.x * 255, integral_color.y * 255, integral_color.z * 255, 255);
    surf2Dwrite(rgba, canvas, x * 4, y);
}

extern "C"
cudaError_t ray_cast_texture(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int width, int height,
    CudaVolumeInfos volume_info, CudaRayCastInfos ray_cast_info, cudaSurfaceObject_t canvas) {
    //1 launch ray cast kernel

    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    
    const int s_mem_size = get_s_array_size(ray_cast_info.label_level);
    int memcpy_step = s_mem_size / BLOCK_SIZE*BLOCK_SIZE;
    if (memcpy_step * BLOCK_SIZE * BLOCK_SIZE != s_mem_size) {
        memcpy_step += 1;
    }

    kernel_ray_cast_main_texture << <grid, block, s_mem_size >> >(entry_tex, exit_tex, width, height, volume_info, ray_cast_info, canvas, memcpy_step);

    return cudaThreadSynchronize();
}

extern "C"
cudaError_t ray_cast_surface(cudaSurfaceObject_t entry_suf, cudaSurfaceObject_t exit_suf, int width, int height,
    CudaVolumeInfos volume_info, CudaRayCastInfos ray_cast_info, cudaSurfaceObject_t canvas) {
    //1 launch ray cast kernel

    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(width / BLOCK_SIZE, height / BLOCK_SIZE);

    const int s_mem_size = get_s_array_size(ray_cast_info.label_level);
    int memcpy_step = s_mem_size / BLOCK_SIZE*BLOCK_SIZE;
    if (memcpy_step * BLOCK_SIZE * BLOCK_SIZE != s_mem_size) {
        memcpy_step += 1;
    }

    kernel_ray_cast_main_surface << <grid, block, s_mem_size >> >(entry_suf, exit_suf, width, height, volume_info, ray_cast_info, canvas, memcpy_step);

    return cudaThreadSynchronize();
}