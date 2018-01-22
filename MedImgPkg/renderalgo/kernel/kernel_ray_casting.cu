#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_functions.h>

#include "../mi_ray_caster_cuda_define.h"


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
    return 72 * label_level;
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

inline __device__ float4* get_mask_overlay_color_array(int label_level) {
    return (float4*)((char*)(s_array) + 56 * label_level);
}

inline __device__ float3 cal_gradient(CudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos_norm) {
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
    //float ksc = 0.0f;
    float alpha = input_color.w*(0.5f + kss * __powf(fn, kse));
    alpha = clamp(alpha, 0.0f, 1.0f);

    output_color = clamp(output_color, 0.0f, 1.0f);
    return make_float4(output_color, alpha);
}

inline __device__ int access_mask_nearest(CudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos) {
    return (int)(tex3D<float>(volume_infos->mask_tex, sample_pos.x, sample_pos.y, sample_pos.z)*255.0f);
}

inline __device__ int access_mask_linear(CudaVolumeInfos* __restrict__ volume_infos, float3 sample_pos) {
    return (int)(tex3D<float>(volume_infos->mask_tex, sample_pos.x, sample_pos.y, sample_pos.z)*255.0f);
}

inline __device__ void composite_dvr(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 sample_pos, float3 ray_dir, float4* __restrict__ integral_color) {
    float3 sample_norm = sample_pos*volume_infos->dim_r;
    //sample_norm = f3_mul_f3(sample_pos,dim3_r);

    int label = 0;
    if (1 == ray_cast_infos->mask_mode) {
        label = access_mask_nearest(volume_infos, sample_norm);
    } else if (0 == ray_cast_infos->mask_mode) {
        label = 0;
    } else {
        label = access_mask_linear(volume_infos, sample_norm);
    }

    if (label == 0 && 0 != ray_cast_infos->mask_mode) {
        return;
    }

    float* s_wl_array = get_s_wl_array(ray_cast_infos->label_level);
    float ww = s_wl_array[label * 2];
    float wl = s_wl_array[label * 2 + 1];
    float min_gray = wl - ww*0.5f;

    float gray = tex3D<float>(volume_infos->volume_tex, sample_norm.x, sample_norm.y, sample_norm.z);
    gray = (gray - min_gray) / ww;
    //gray = clamp(gray,0.0f,1.0f);
    cudaTextureObject_t* color_opacity_texture_array = get_s_color_opacity_texture_array(ray_cast_infos->label_level);
    float4 color_ori = tex1D<float4>(color_opacity_texture_array[label], gray + ray_cast_infos->color_opacity_texture_shift);
    
    float alpha;
    if (color_ori.w > 0.0f) {
        if (1 == ray_cast_infos->shading_mode) {
            color_ori = shade_phong(volume_infos, ray_cast_infos, sample_pos, sample_norm, ray_dir, color_ori, label);
        }
        
        alpha = (1 - __powf(1 - color_ori.w, ray_cast_infos->opacity_correction))*(1.0f - integral_color->w);
        integral_color->x += color_ori.x * alpha;
        integral_color->y += color_ori.y * alpha;
        integral_color->z += color_ori.z * alpha;
        integral_color->w += alpha;
    }
}

__device__ float4 kernel_ray_cast_dvr(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos, 
    float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color, float3* ray_end) {
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
    *ray_end = sample_pos;
    return integral_color;
}

__device__ float4 kernel_ray_cast_mip(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color, float* mip_gray, float3* mip_pos) {
    float4 integral_color = input_color;
    float3 sample_pos, sample_norm;
    int label = 0;
    float max_gray = -65535.0f;
    float3 max_pos;
    //int max_label = 0;
    float gray = 0.0f;
    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        sample_norm = sample_pos*volume_infos->dim_r;

        label = 0;
        if (1 == ray_cast_infos->mask_mode) {
            label = access_mask_nearest(volume_infos, sample_norm);
        } else if (0 == ray_cast_infos->mask_mode) {
            label = 0;
        } else {
            label = access_mask_linear(volume_infos, sample_norm);
        }

        if (label == 0 && 0 != ray_cast_infos->mask_mode) {
            continue;
        }

        gray = tex3D<float>(volume_infos->volume_tex, sample_norm.x, sample_norm.y, sample_norm.z);
        if (gray > max_gray) {
            max_gray = gray;
            max_pos = sample_pos;
            //max_label = label;
        }
    }

    if (max_gray < -65535.0f + 1.0f) {
        *mip_gray = 0.0f;
        *mip_pos = make_float3(0.0f);
        return integral_color;
    } else {
        *mip_gray = max_gray;
        *mip_pos = max_pos;
    }

    float ww = ray_cast_infos->global_ww;
    float wl = ray_cast_infos->global_wl;
    gray = (*mip_gray - wl + ww*0.5f) / ww;

    //TODO 
    //1 blend with graphic
    //2 mip pos and mip label
    integral_color = tex1D<float4>(ray_cast_infos->pseudo_color_texture, gray + ray_cast_infos->pseudo_color_texture_shift);

    return integral_color;
}

__device__ float4 kernel_ray_cast_minip(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color, float* mip_gray, float3* mip_pos) {
    float4 integral_color = input_color;
    float3 sample_pos, sample_norm;
    int label = 0;
    float min_gray = 65535.0f;
    float3 min_pos;
    //int min_label = 0;
    float gray = 0.0f;
    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        sample_norm = sample_pos*volume_infos->dim_r;

        label = 0;
        if (1 == ray_cast_infos->mask_mode) {
            label = access_mask_nearest(volume_infos, sample_norm);
        }
        else if (0 == ray_cast_infos->mask_mode) {
            label = 0;
        }
        else {
            label = access_mask_linear(volume_infos, sample_norm);
        }

        if (label == 0 && 0 != ray_cast_infos->mask_mode) {
            continue;
        }

        gray = tex3D<float>(volume_infos->volume_tex, sample_norm.x, sample_norm.y, sample_norm.z);
        if (gray > ray_cast_infos->minip_threshold && gray < min_gray) {
            min_gray = gray;
            min_pos = sample_pos;
            //min_label = label;
        }
    }

    if (min_gray > 65535.0f - 1.0f) {
        *mip_gray = 0.0f;
        *mip_pos = make_float3(0.0f);
        return integral_color;
    }
    else {
        *mip_gray = min_gray;
        *mip_pos = min_pos;
    }

    float ww = ray_cast_infos->global_ww;
    float wl = ray_cast_infos->global_wl;
    gray = (*mip_gray - wl + ww*0.5f) / ww;

    //TODO 
    //1 blend with graphic
    //2 mip pos and mip label
    integral_color = tex1D<float4>(ray_cast_infos->pseudo_color_texture, gray + ray_cast_infos->pseudo_color_texture_shift);

    return integral_color;
}

__device__ float4 kernel_ray_cast_average(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color, float* mip_gray) {
    float4 integral_color = input_color;
    float3 sample_pos, sample_norm;
    int label = 0;
    float gray_sum = 0.0f;
    float gray_count = 0.0f;
    float gray = 0.0f;
    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        sample_norm = sample_pos*volume_infos->dim_r;

        label = 0;
        if (1 == ray_cast_infos->mask_mode) {
            label = access_mask_nearest(volume_infos, sample_norm);
        }
        else if (0 == ray_cast_infos->mask_mode) {
            label = 0;
        }
        else {
            label = access_mask_linear(volume_infos, sample_norm);
        }

        if (label == 0 && 0 != ray_cast_infos->mask_mode) {
            continue;
        }

        gray = tex3D<float>(volume_infos->volume_tex, sample_norm.x, sample_norm.y, sample_norm.z);
        gray_sum += gray;
        gray_count += 1.0f;
    }

    if (gray_count > 0.0f) {
        *mip_gray = gray_sum / gray_count;
    } else {
        *mip_gray = 0.0f;
        return integral_color;
    }

    float ww = ray_cast_infos->global_ww;
    float wl = ray_cast_infos->global_wl;
    gray = (*mip_gray - wl + ww*0.5f) / ww;

    //TODO 
    //1 blend with graphic
    //2 mip pos and mip label
    integral_color = tex1D<float4>(ray_cast_infos->pseudo_color_texture, gray + ray_cast_infos->pseudo_color_texture_shift);

    return integral_color;
}

//Encoding label to intger array 4*32 can contain 0~127 labels
__device__ void label_encode(int label, int4 *mask_flag) {
    if (label < 32) {
        (*mask_flag).x = (*mask_flag).x | (1 << label);
    } else if (label < 64) {
        (*mask_flag).y = (*mask_flag).y | (1 << (label - 32));
    } else if (label < 96) {
        (*mask_flag).z = (*mask_flag).z | (1 << (label - 64));
    } else {
        (*mask_flag).w = (*mask_flag).w | (1 << (label - 96));
    }
}

//Decoding label from intger array 4*32 can contain 0~127 labels
__device__ bool label_decode(int label, int4 *mask_flag) {
    bool is_hitted = false;
    if (label < 32) {
        is_hitted = ((1 << label) & (*mask_flag).x) != 0;
    } else if (label < 64) {
        is_hitted = ((1 << (label - 32)) & (*mask_flag).y) != 0;
    } else if (label < 96) {
        is_hitted = ((1 << (label - 64)) & (*mask_flag).z) != 0;
    } else {
        is_hitted = ((1 << (label - 96)) & (*mask_flag).w) != 0;
    }
    return is_hitted;
}

__device__ float4 mask_overlay(CudaVolumeInfos* __restrict__ volume_infos, CudaRayCastInfos* __restrict__ ray_cast_infos,
    float3 ray_dir, float3 ray_start, float start_step, float end_step, float4 input_color) {
    float4 integral_color = input_color;
    float3 sample_pos, sample_norm;
    int label = 0;
    int4 tracing_label_code = make_int4(0);
    float overlay_opacity = ray_cast_infos->mask_overlay_opacity;

    //tracing & coding label(no sequence)
    for (float i = start_step; i < end_step; i += 1.0f) {
        sample_pos = ray_start + ray_dir*i;
        sample_norm = sample_pos*volume_infos->dim_r;

        label = access_mask_nearest(volume_infos, sample_norm);

        if (label != 0) {
            label_encode(label, &tracing_label_code);
        }
    }

    //blending traced label to pre-color
    float4 label_color;
    int* visible_label = get_s_visible_label_array(ray_cast_infos->label_level);
    for (int i = 0; i < ray_cast_infos->label_level; ++i) {
        if (visible_label[i] != 0) {
            if (label_decode(i, &tracing_label_code)) {
                label_color = get_mask_overlay_color_array(ray_cast_infos->label_level)[i];
                integral_color.x = (1.0f - overlay_opacity)* integral_color.x + overlay_opacity * label_color.x;
                integral_color.y = (1.0f - overlay_opacity)* integral_color.y + overlay_opacity * label_color.y;
                integral_color.z = (1.0f - overlay_opacity)* integral_color.z + overlay_opacity * label_color.z;
            }
        }
    }

    integral_color.w = 1.0f;
    return integral_color;
}

__device__ int kernel_preprocess(uint x, uint y, float3 entry, float3 exit, CudaRayCastInfos* __restrict__ ray_cast_infos,  float3* __restrict__ ray_start, float3* __restrict__ ray_dir, float* __restrict__ start_step, float* __restrict__ end_step) {
    float3 ray_dir0 = exit - entry;
    float3 ray_dir_norm = normalize(ray_dir0);
    float ray_length = length(ray_dir0);
    if (ray_length < 1e-5f) {
        return -1;
    }

    float sample_step = ray_cast_infos->sample_step;

    float adjust = 0.0f;
    if (1 == ray_cast_infos->ray_align_to_view_plane) {
        float len = dot(entry - ray_cast_infos->eye_position, ray_dir_norm);
        adjust = len / sample_step;
        adjust = (ceilf(adjust) - adjust)*sample_step;
        ray_length = ray_length - adjust;
        if (ray_length < 1e-5f) {
            return -1;    
        }
    }

    *ray_start = entry + adjust*ray_dir_norm;

    if (1 == ray_cast_infos->jittering) {
        const float random_size_r = 0.03125f;// 1/32        
        float r = tex2D<float>(ray_cast_infos->random_texture, x * random_size_r, y * random_size_r);
        *ray_start += r*ray_dir_norm;
    }

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

    /*if (thread_idx == 0) {
        char* dst = (char*)s_array;
        char* src = (char*)d_mapped_array;
        for (int i = 0; i < size; ++i) {
            dst[i] = src[i];
        }
    }*/
}

__global__ void kernel_ray_cast_main_texture(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int2 entry_exit_size,
    CudaVolumeInfos volume_infos, CudaRayCastInfos ray_cast_infos, cudaSurfaceObject_t canvas, cudaSurfaceObject_t ray_end_canvas, int2 canvas_size, int quarter, int memcpy_step) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    //fill shared array 
    uint local_thread = threadIdx.y * blockDim.x + threadIdx.x;
    fill_shared_array(local_thread, ray_cast_infos.d_shared_mapped_memory, get_s_array_size(ray_cast_infos.label_level), memcpy_step);
    __syncthreads();

    if (x > canvas_size.x - 1 || y > canvas_size.y - 1) {
        return;
    }

    float4 entry, exit;
    if (0 == quarter) {
        entry = tex2D<float4>(entry_tex, x, y);
        exit = tex2D<float4>(exit_tex, x, y);
    } else {
        entry = tex2D<float4>(entry_tex, x<<1, y<<1);
        exit = tex2D<float4>(exit_tex, x<<1, y<<1);
    }

    float3 entry3 = make_float3(entry);
    float3 exit3  = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    if (1 == ray_cast_infos.test_code) {
        uchar4 rgba_entry = make_uchar4(entry.x / volume_infos.dim.x * 255, entry.y / volume_infos.dim.y * 255, entry.z / volume_infos.dim.z * 255, 255);
        surf2Dwrite(rgba_entry, canvas, x << 2, y);
        return;
    } else if (2 == ray_cast_infos.test_code) {
        uchar4 rgba_exit = make_uchar4(exit.x / volume_infos.dim.x * 255, exit.y / volume_infos.dim.y * 255, exit.z / volume_infos.dim.z * 255, 255);
        surf2Dwrite(rgba_exit, canvas, x << 2, y);
        return;
    }

    if (0 != kernel_preprocess(x, y, entry3, exit3, &ray_cast_infos, &ray_start, &ray_dir, &start_step, &end_step)) {
        uchar4 rgba = make_uchar4(0,0,0,0);
        surf2Dwrite(rgba, canvas, x << 2, y);
        return;
    }

    float4 input_color = make_float4(0.0f);
    float4 integral_color = make_float4(0.0f);
    float  mip_gray = 0.0f;
    float3 ray_end = make_float3(0.0f);
    if (0 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_dvr(&volume_infos, &ray_cast_infos, 
            ray_dir, ray_start, start_step, end_step, input_color, &ray_end);
    } else if (1 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_mip(&volume_infos, &ray_cast_infos, 
            ray_dir, ray_start, start_step, end_step, input_color, &mip_gray, &ray_end);
    } else if (2 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_minip(&volume_infos, &ray_cast_infos, 
            ray_dir, ray_start, start_step, end_step, input_color, &mip_gray, &ray_end);
    } else if (3 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_average(&volume_infos, &ray_cast_infos, 
            ray_dir, ray_start, start_step, end_step, input_color, &mip_gray);
    }

    if (0 != ray_cast_infos.mask_overlay_mode) {
        integral_color = mask_overlay(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, integral_color);
    }
     
    clamp(integral_color, 0.0f, 1.0f);

    uchar4 rgba = make_uchar4(integral_color.x * 255, integral_color.y * 255 , integral_color.z * 255, 255);
    surf2Dwrite(rgba, canvas, x << 2 , y);

    if(3 != ray_cast_infos.composite_mode && 0 != ray_end_canvas) {
        uchar4 ray_end_rgba = make_uchar4(
            (unsigned char)(ray_end.x*volume_infos.dim_r.x*255.0f), 
            (unsigned char)(ray_end.y*volume_infos.dim_r.y*255.0f), 
            (unsigned char)(ray_end.z*volume_infos.dim_r.z*255.0f), 255);
        surf2Dwrite(ray_end_rgba, ray_end_canvas, x << 2, y);
    }
}

__global__ void kernel_ray_cast_main_surface(cudaSurfaceObject_t entry_surf, cudaSurfaceObject_t exit_surf, int2 entry_exit_size,
    CudaVolumeInfos volume_infos, CudaRayCastInfos ray_cast_infos, cudaSurfaceObject_t canvas, cudaSurfaceObject_t ray_end_canvas, int2 canvas_size, int quarter, int memcpy_step) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    //fill shared array
    uint local_thread = threadIdx.y * blockDim.x + threadIdx.x;
    fill_shared_array(local_thread, ray_cast_infos.d_shared_mapped_memory, get_s_array_size(ray_cast_infos.label_level), memcpy_step);
    __syncthreads();

    if (x > canvas_size.x - 1 || y > canvas_size.y - 1) {
        return;
    }

    float4 entry, exit;
    if (0 == quarter) {
        surf2Dread(&entry, entry_surf, x << 4, y);
        surf2Dread(&exit, exit_surf, x << 4, y);
    } else {
        surf2Dread(&entry, entry_surf, x << 5, y << 1);
        surf2Dread(&exit, exit_surf, x << 5, y << 1);
    }
    
    float3 entry3 = make_float3(entry);
    float3 exit3 = make_float3(exit);

    float3 ray_start, ray_dir;
    float start_step, end_step;

    if (1 == ray_cast_infos.test_code) {
        uchar4 rgba_entry= make_uchar4(entry.x / volume_infos.dim.x * 255, entry.y / volume_infos.dim.y * 255, entry.z / volume_infos.dim.z * 255, 255);
        surf2Dwrite(rgba_entry, canvas, x << 2, y);
        return;
    } else if (2 == ray_cast_infos.test_code) {
        uchar4 rgba_exit= make_uchar4(exit.x / volume_infos.dim.x * 255, exit.y / volume_infos.dim.y * 255, exit.z / volume_infos.dim.z * 255, 255);
        surf2Dwrite(rgba_exit, canvas, x << 2, y);
        return;
    }

    if (0 != kernel_preprocess(x, y, entry3, exit3, &ray_cast_infos, &ray_start, &ray_dir, &start_step, &end_step)) {
        uchar4 rgba = make_uchar4(0, 0, 0, 0);
        surf2Dwrite(rgba, canvas, x << 2, y);
        return;
    }

    float4 input_color = make_float4(0.0f);
    float4 integral_color = make_float4(0.0f);
    float  mip_gray = 0.0f;
    float3 ray_end = make_float3(0.0f);
    if (0 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_dvr(&volume_infos, &ray_cast_infos,
            ray_dir, ray_start, start_step, end_step, input_color, &ray_end);
    } else if (1 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_mip(&volume_infos, &ray_cast_infos,
            ray_dir, ray_start, start_step, end_step, input_color, &mip_gray, &ray_end);
    } else if (2 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_minip(&volume_infos, &ray_cast_infos,
            ray_dir, ray_start, start_step, end_step, input_color, &mip_gray, &ray_end);
    } else if (3 == ray_cast_infos.composite_mode) {
        integral_color = kernel_ray_cast_average(&volume_infos, &ray_cast_infos,
            ray_dir, ray_start, start_step, end_step, input_color, &mip_gray);
    }

    if (0 != ray_cast_infos.mask_overlay_mode) {
        integral_color = mask_overlay(&volume_infos, &ray_cast_infos, ray_dir, ray_start, start_step, end_step, integral_color);
    }

    clamp(integral_color, 0.0f, 1.0f);

    uchar4 rgba = make_uchar4(integral_color.x * 255, integral_color.y * 255, integral_color.z * 255, 255);
    surf2Dwrite(rgba, canvas, x << 2, y);

    if(3 != ray_cast_infos.composite_mode && 0 != ray_end_canvas) {
        uchar4 ray_end_rgba = make_uchar4(
            (unsigned char)(ray_end.x*volume_infos.dim_r.x*255.0f), 
            (unsigned char)(ray_end.y*volume_infos.dim_r.y*255.0f), 
            (unsigned char)(ray_end.z*volume_infos.dim_r.z*255.0f), 255);
        surf2Dwrite(ray_end_rgba, ray_end_canvas, x << 2, y);
    }
}

__global__ void kernel_quarter_map_back(cudaSurfaceObject_t quater_canvas, int2 quarter_size, cudaSurfaceObject_t canvas, int2 canvas_size) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > canvas_size.x - 1 || y > canvas_size.y - 1) {
        return;
    }

    float2 tex_coord = make_float2(x / (float)canvas_size.x, y / (float)canvas_size.y);
    float2 src_coord = tex_coord * make_float2(quarter_size);
    float2 src_coord00 = floorf(src_coord);

    int2 s_00 = make_int2(src_coord00);
    int2 s_10 = s_00 + make_int2(1, 0);
    int2 s_01 = s_00 + make_int2(0, 1);
    int2 s_11 = s_00 + make_int2(1, 1);

    s_10 = clamp(s_10, make_int2(0, 0), quarter_size - 1);
    s_01 = clamp(s_10, make_int2(0, 0), quarter_size - 1);
    s_11 = clamp(s_10, make_int2(0, 0), quarter_size - 1);
    
    uchar4 v00, v01, v10, v11;
    surf2Dread(&v00, quater_canvas, s_00.x << 2, s_00.y);
    surf2Dread(&v01, quater_canvas, s_01.x << 2, s_01.y);
    surf2Dread(&v10, quater_canvas, s_10.x << 2, s_10.y);
    surf2Dread(&v11, quater_canvas, s_11.x << 2, s_11.y);

    float dx1 = src_coord.x - src_coord00.x;
    float dx0 = 1.0f - dx1;
    float dy1 = src_coord.y - src_coord00.y;
    float dy0 = 1.0f - dy1;

    float4 v_x0 = make_float4(v00.x, v00.y, v00.z, v00.w) * dx0 + make_float4(v10.x, v10.y, v10.z, v10.w) * dx1;
    float4 v_x1 = make_float4(v01.x, v01.y, v01.z, v01.w) * dx0 + make_float4(v11.x, v11.y, v11.z, v11.w) * dx1;
    float4 rgba32f = v_x0 * dy0 + v_x1 * dy1;
    rgba32f = clamp(rgba32f, make_float4(0.0f), make_float4(255.0f));
    uchar4 rgba8 = make_uchar4(rgba32f.x, rgba32f.y, rgba32f.z, rgba32f.w);

    surf2Dwrite(rgba8, canvas, x << 2 , y);
}

extern "C"
cudaError_t ray_cast_texture(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int2 entry_exit_size,
    CudaVolumeInfos volume_info, CudaRayCastInfos ray_cast_info, cudaSurfaceObject_t canvas, cudaSurfaceObject_t ray_end_canvas, int2 canvas_size, int quarter) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(canvas_size.x / BLOCK_SIZE, canvas_size.y / BLOCK_SIZE);
    if (grid.x * BLOCK_SIZE != canvas_size.x) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZE != canvas_size.y) {
        grid.y += 1;
    }
    
    const int s_mem_size = get_s_array_size(ray_cast_info.label_level);
    int memcpy_step = s_mem_size / (BLOCK_SIZE*BLOCK_SIZE);
    if (memcpy_step * (BLOCK_SIZE*BLOCK_SIZE) != s_mem_size) {
        memcpy_step += 1;
    }

    kernel_ray_cast_main_texture << <grid, block, s_mem_size >> >(entry_tex, exit_tex, entry_exit_size, volume_info, ray_cast_info, canvas, ray_end_canvas, canvas_size, quarter, memcpy_step);

    return cudaThreadSynchronize();
}

extern "C"
cudaError_t ray_cast_surface(cudaSurfaceObject_t entry_suf, cudaSurfaceObject_t exit_suf, int2 entry_exit_size,
    CudaVolumeInfos volume_info, CudaRayCastInfos ray_cast_info, cudaSurfaceObject_t canvas, cudaSurfaceObject_t ray_end_canvas, int2 canvas_size, int quarter) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(canvas_size.x / BLOCK_SIZE, canvas_size.y / BLOCK_SIZE);
    if (grid.x * BLOCK_SIZE != canvas_size.x) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZE != canvas_size.y) {
        grid.y += 1;
    }

    const int s_mem_size = get_s_array_size(ray_cast_info.label_level);
    int memcpy_step = s_mem_size / BLOCK_SIZE*BLOCK_SIZE;
    if (memcpy_step * BLOCK_SIZE * BLOCK_SIZE != s_mem_size) {
        memcpy_step += 1;
    }

    kernel_ray_cast_main_surface << <grid, block, s_mem_size >> >(entry_suf, exit_suf, entry_exit_size, volume_info, ray_cast_info, canvas, ray_end_canvas, canvas_size, quarter, memcpy_step);

    return cudaThreadSynchronize();
}

extern "C"
cudaError_t quarter_map_back(cudaSurfaceObject_t quarter_canvas, int2 quarter_size, cudaSurfaceObject_t canvas, int2 canvas_size) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(canvas_size.x / BLOCK_SIZE, canvas_size.y / BLOCK_SIZE);
    if (grid.x * BLOCK_SIZE != canvas_size.x) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZE != canvas_size.y) {
        grid.y += 1;
    }

    kernel_quarter_map_back<<<grid, block>>>(quarter_canvas, quarter_size, canvas, canvas_size);

    return cudaThreadSynchronize();
}