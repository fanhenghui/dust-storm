#ifndef MEDIMG_UT_MI_CUDA_VR_COMMON_H
#define MEDIMG_UT_MI_CUDA_VR_COMMON_H

#ifdef WIN32
#include "gl/glew.h"
#else
#include <GL/glew.h>
#endif

#include <stdio.h>
#include <iostream>
#include <list>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <device_functions.h>

#include "mi_cuda_graphic.h"
#include "med_img_pkg_config.h"

#define CHECK_CUDA_ERROR {\
cudaError_t err = cudaGetLastError(); \
if (err != cudaSuccess) {\
    std::cout << "CUDA error: " << err << " in function: " << __FUNCTION__ <<\
    " line: " << __LINE__ << std::endl; \
}}\

#define MAX_MASK_LEVEL 128

class CudaMemShield {
public:
    CudaMemShield() {

    }
    
    ~CudaMemShield() {
        for (auto it = _shields.begin(); it != _shields.end(); ++it) {
            if (*it) {
                cudaFree(*it);
            }
        }
        _shields.clear();
    }
    
    void add_shield(void* mem) {
        if (mem) {
            _shields.push_back(mem);
        }
        
    }

    void remove_shield(void* mem) {
        for (auto it = _shields.begin(); it != _shields.end(); ++it) {
            if ((*it) == mem) {
                _shields.erase(it);
                break;
            }
        }
    }
private:
    std::list<void*> _shields;
private:
    DISALLOW_COPY_AND_ASSIGN(CudaMemShield);
};



struct cudaGLTextureWriteOnly {
    GLuint gl_tex_id;
    GLenum target;
    cudaGraphicsResource *cuda_res;
    cudaArray* d_cuda_array;

    cudaGLTextureWriteOnly() {
        gl_tex_id = 0;
        target = GL_TEXTURE_1D;
        cuda_res = NULL;
        d_cuda_array = NULL;
    }
};

inline  __host__ void register_image(cudaGLTextureWriteOnly& cuda_tex) {
    cudaGraphicsGLRegisterImage(&cuda_tex.cuda_res, cuda_tex.gl_tex_id, cuda_tex.target, cudaGraphicsRegisterFlagsWriteDiscard);
    CHECK_CUDA_ERROR;
}

inline __host__ void map_image(cudaGLTextureWriteOnly& cuda_tex) {
    cuda_tex.d_cuda_array = NULL;
    cudaGraphicsMapResources(1, &cuda_tex.cuda_res);
    cudaGraphicsSubResourceGetMappedArray(&cuda_tex.d_cuda_array, cuda_tex.cuda_res, 0, 0);
}

inline __host__ void write_image(cudaGLTextureWriteOnly& cuda_tex, void* d_buffer, size_t count) {
    if (cuda_tex.d_cuda_array) {
        cudaMemcpyToArray(cuda_tex.d_cuda_array, 0, 0, d_buffer, count, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR;
    }
}

inline  __host__  void release_image(cudaGLTextureWriteOnly& cuda_tex) {
    cudaGraphicsUnregisterResource(cuda_tex.cuda_res);
    cuda_tex.cuda_res = NULL;
}

inline __host__ void unmap_image(cudaGLTextureWriteOnly& cuda_tex) {
    cudaGraphicsUnmapResources(1, &cuda_tex.cuda_res);
}

struct cudaGLTextureReadOnly {
    GLuint gl_tex_id;
    GLenum target;
    cudaTextureObject_t cuda_tex_obj;//GLTexture2D->cudaTextureType2D
    cudaGraphicsResource *cuda_res;
    cudaArray* d_cuda_array;

    cudaGLTextureReadOnly() {
        gl_tex_id = 0;
        target = GL_TEXTURE_1D;
        cuda_tex_obj = NULL;
        cuda_res = NULL;
        d_cuda_array = NULL;
    }
};

inline  __host__ void register_image(cudaGLTextureReadOnly& cuda_tex) {
    cudaGraphicsGLRegisterImage(&cuda_tex.cuda_res, cuda_tex.gl_tex_id, cuda_tex.target, cudaGraphicsRegisterFlagsReadOnly);
    CHECK_CUDA_ERROR;
}

inline __host__ void bind_texture(cudaGLTextureReadOnly& cuda_tex, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, bool normalized_coords) {
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_tex.d_cuda_array;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeMirror;
    tex_desc.addressMode[1] = cudaAddressModeMirror;
    tex_desc.filterMode = filter_mode;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = normalized_coords;

    cudaCreateTextureObject(&cuda_tex.cuda_tex_obj, &res_desc, &tex_desc, NULL);
}

inline __host__ void map_image(cudaGLTextureReadOnly& cuda_tex) {
    cuda_tex.d_cuda_array = NULL;
    cudaGraphicsMapResources(1, &cuda_tex.cuda_res);
    cudaGraphicsSubResourceGetMappedArray(&cuda_tex.d_cuda_array, cuda_tex.cuda_res, 0, 0);
}

inline  __host__  void release_image(cudaGLTextureReadOnly& cuda_tex) {
    cudaGraphicsUnregisterResource(cuda_tex.cuda_res);
    cudaDestroyTextureObject(cuda_tex.cuda_tex_obj);
    cuda_tex.cuda_res = NULL;
}

inline __host__ void unmap_image(cudaGLTextureReadOnly& cuda_tex) {
    cudaGraphicsUnmapResources(1, &cuda_tex.cuda_res);
}

struct cudaRayCastInfos {
    float sample_step;
    int mask_level;//1(non-mask)->8 ->16 ->32 ->64 ->128
    cudaArray* d_lut_array;
    cudaTextureObject_t lut_tex_obj[MAX_MASK_LEVEL];//cudaTextureType1DArray
    mat4 mat_normal;//transpose(inverse(mat_m2v))
    float3 light_position;
    float3 ambient_color;//RGB norm
    float ambient_intensity;//intensity
    float lut_length;//length of one CUDA texture 1D
    float* d_wl_array;//mask_level * 2
    float* d_material_array;//mask_level * 9float {diffuseRGB&intensity ; sepcular RGB&intensity&shininess}

    cudaRayCastInfos() {
        sample_step = 0.5f;
        mask_level = 1;
        d_lut_array = NULL;
        //lut_tex_obj = NULL;
        memset(lut_tex_obj , 0 , sizeof(cudaTextureObject_t)*MAX_MASK_LEVEL);
        lut_length = 512;
        d_wl_array = NULL;
        d_material_array = NULL;
        ambient_color = make_float3(1.0f,1.0f,1.0f);
        ambient_intensity = 0.3f;
    }
};

struct cudaVolumeInfos
{
    cudaArray* d_volume_array;
    cudaTextureObject_t volume_tex_obj;//cudaTextureType3D
    cudaArray* d_mask_array;
    cudaTextureObject_t mask_tex_obj;//cudaTextureType3D
    uint3 dim;
    float3 dim_r;
    float3 sample_shift;

    cudaVolumeInfos() {
        d_volume_array = NULL;
        volume_tex_obj = NULL;
        d_mask_array = NULL;
        mask_tex_obj = NULL;
    }
};



#endif