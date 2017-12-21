#ifndef MEDIMG_UT_MI_CUDA_VR_H
#define MEDIMG_UT_MI_CUDA_VR_H

#ifdef WIN32
#include "gl/glew.h"
#else
#include <GL/glew.h>
#endif

#include <stdio.h>
#include <vector_types.h>

struct cudaGLTexture {
    GLuint gl_tex_id;
    GLenum target;
    int width;
    int height;
    cudaTextureObject_t cuda_tex_obj;//GLTexture2D->cudaTextureType2D
    cudaGraphicsResource *cuda_res;
    cudaArray* d_cuda_array;
};

struct cudaRayCastInfos {
    float sample_step;
    int mask_level;//8 ->16 ->32 ->64 ->128
    cudaArray* d_lut_array;
    cudaTextureObject_t lut_tex_obj;//cudaTextureType1DArray
    float lut_length;//lenth of one CUDA texture 1D
    float* d_wl_array;//mask_level * 2
    float* d_material_array;//mask_level * 2
};

struct cudaVolumeInfos 
{
    cudaArray* d_volume_array;
    cudaTextureObject_t volume_tex_obj;//cudaTextureType3D
    cudaArray* d_mask_array;
    cudaTextureObject_t mask_tex_obj;//cudaTextureType3D
    uint3 dim;
};

#endif