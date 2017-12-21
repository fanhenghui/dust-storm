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
    cudaTextureObject_t cuda_tex_obj;
    cudaGraphicsResource *cuda_res;
    cudaArray* cuda_array;
};

struct cudaRayCastInfos {
    float sample_step;
    cudaArray* lut_array;
    cudaTextureObject_t lut_tex_obj;//后续需要考虑Array
    float lut_length;
    float WL;
};

struct cudaVolumeInfos 
{
    cudaArray* volume_array;
    cudaTextureObject_t volume_tex_obj;
    cudaArray* mask_array;
    cudaTextureObject_t mask_tex_obj;
    uint3 dim;
};

#endif