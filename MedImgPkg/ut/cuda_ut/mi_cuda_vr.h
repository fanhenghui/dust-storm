#ifndef MEDIMG_UT_MI_CUDA_VR_H
#define MEDIMG_UT_MI_CUDA_VR_H

#ifdef WIN32
#include "gl/glew.h"
#else
#include <GL/glew.h>
#endif

#include <stdio.h>
#include <GL/glew.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

#include "arithmetic/mi_cuda_math.h"
#include "arithmetic/mi_matrix4.h"

#define CHECK_CUDA_ERROR {\
cudaError_t err = cudaGetLastError(); \
if (err != cudaSuccess) {\
    std::cout << "CUDA error: " << err << " in function: " << __FUNCTION__ <<\
    " line: " << __LINE__ << std::endl; \
}}\

struct cudaGLTextureWriteOnly {
    GLuint gl_tex_id;
    GLenum target;
    cudaGraphicsResource *cuda_res;
    cudaArray* d_cuda_array;    
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

//struct cudaGLTextureReadOnly {
//    GLuint gl_tex_id;
//    GLenum target;
//    cudaTextureObject_t cuda_tex_obj;//GLTexture2D->cudaTextureType2D
//    cudaGraphicsResource *cuda_res;
//    cudaArray* d_cuda_array;
//
//    __host__ cudaGLTextureReadOnly() :gl_tex_id(0), target(GL_TEXTURE_2D), cuda_res(NULL), d_cuda_array(NULL) {
//
//    }
//
//    __host__ void register_image() {
//        cuda_res = NULL;
//        cudaGraphicsGLRegisterImage(&cuda_res, gl_tex_id, target, cudaGraphicsRegisterFlagsReadOnly);
//        CHECK_CUDA_ERROR;
//    }
//
//    __host__ void map() {
//        d_cuda_array = NULL;
//        cudaGraphicsMapResources(1, &cuda_res);
//        cudaGraphicsSubResourceGetMappedArray(&d_cuda_array, cuda_res, 0, 0);
//    }
//
//    __host__ void unmap() {
//        cudaGraphicsUnmapResources(1, &cuda_res);
//    }
//};

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
    float lut_length;//length of one CUDA texture 1D
    float* d_wl_array;//mask_level * 2
    float* d_material_array;//mask_level * 2 {float4(diffuse) float4(specular) float4(shininess reserve0/1/2)}
};

struct cudaVolumeInfos 
{
    cudaArray* d_volume_array;
    cudaTextureObject_t volume_tex_obj;//cudaTextureType3D
    cudaArray* d_mask_array;
    cudaTextureObject_t mask_tex_obj;//cudaTextureType3D
    uint3 dim;
};

struct Viewport {
    int x, y, width, height;
    __host__ __device__ Viewport(int x_, int y_, int width_, int height_) : x(x_), y(y_), width(width_), height(height_) {}
};

//Math
#define INF     2e10f

struct mat4 {
    float4 col0, col1, col2, col3;
    __host__ __device__ mat4(float4 col0_, float4 col1_, float4 col2_, float4 col3_) :col0(col0_), col1(col1_), col2(col2_), col3(col3_) {
    }
};

struct Rectangle {
    float3 p0, p1, p2, p3;//position world coordinate
    float2 c0, c1, c2, c3;//texture coordinate
    __host__ __device__  Rectangle(float3 p0_, float3 p1_, float3 p2_, float3 p3_,
        float2 c0_, float2 c1_, float2 c2_, float2 c3_) :
        p0(p0_), p1(p1_), p2(p2_), p3(p3_),
        c0(c0_), c1(c1_), c2(c2_), c3(c3_) {
    }
};

inline mat4 matrix4_to_mat4(const medical_imaging::Matrix4& mat44) {
    return mat4(
        make_float4(mat44._m[0], mat44._m[1], mat44._m[2], mat44._m[3]),
        make_float4(mat44._m[4], mat44._m[5], mat44._m[6], mat44._m[7]),
        make_float4(mat44._m[8], mat44._m[9], mat44._m[10], mat44._m[11]),
        make_float4(mat44._m[12], mat44._m[13], mat44._m[14], mat44._m[15]));
}

inline __host__ __device__ mat4 operator*(mat4 &m0, mat4 &m1) {
    float4 col0 = make_float4(
        m0.col0.x * m1.col0.x + m0.col1.x * m1.col0.y + m0.col2.x * m1.col0.z + m0.col3.x * m1.col0.w,
        m0.col0.y * m1.col0.x + m0.col1.y * m1.col0.y + m0.col2.y * m1.col0.z + m0.col3.y * m1.col0.w,
        m0.col0.z * m1.col0.x + m0.col1.z * m1.col0.y + m0.col2.z * m1.col0.z + m0.col3.z * m1.col0.w,
        m0.col0.w * m1.col0.x + m0.col1.w * m1.col0.y + m0.col2.w * m1.col0.z + m0.col3.w * m1.col0.w);

    float4 col1 = make_float4(
        m0.col0.x * m1.col1.x + m0.col1.x * m1.col1.y + m0.col2.x * m1.col1.z + m0.col3.x * m1.col1.w,
        m0.col0.y * m1.col1.x + m0.col1.y * m1.col1.y + m0.col2.y * m1.col1.z + m0.col3.y * m1.col1.w,
        m0.col0.z * m1.col1.x + m0.col1.z * m1.col1.y + m0.col2.z * m1.col1.z + m0.col3.z * m1.col1.w,
        m0.col0.w * m1.col1.x + m0.col1.w * m1.col1.y + m0.col2.w * m1.col1.z + m0.col3.w * m1.col1.w);

    float4 col2 = make_float4(
        m0.col0.x * m1.col2.x + m0.col1.x * m1.col2.y + m0.col2.x * m1.col2.z + m0.col3.x * m1.col2.w,
        m0.col0.y * m1.col2.x + m0.col1.y * m1.col2.y + m0.col2.y * m1.col2.z + m0.col3.y * m1.col2.w,
        m0.col0.z * m1.col2.x + m0.col1.z * m1.col2.y + m0.col2.z * m1.col2.z + m0.col3.z * m1.col2.w,
        m0.col0.w * m1.col2.x + m0.col1.w * m1.col2.y + m0.col2.w * m1.col2.z + m0.col3.w * m1.col2.w);

    float4 col3 = make_float4(
        m0.col0.x * m1.col3.x + m0.col1.x * m1.col3.y + m0.col2.x * m1.col3.z + m0.col3.x * m1.col3.w,
        m0.col0.y * m1.col3.x + m0.col1.y * m1.col3.y + m0.col2.y * m1.col3.z + m0.col3.y * m1.col3.w,
        m0.col0.z * m1.col3.x + m0.col1.z * m1.col3.y + m0.col2.z * m1.col3.z + m0.col3.z * m1.col3.w,
        m0.col0.w * m1.col3.x + m0.col1.w * m1.col3.y + m0.col2.w * m1.col3.z + m0.col3.w * m1.col3.w);

    return mat4(col0, col1, col2, col3);
}

inline __host__ __device__ float4 operator*(mat4 &m0, float4 v) {
    return make_float4(
        m0.col0.x * v.x + m0.col1.x * v.y + m0.col2.x * v.z + m0.col3.x * v.w,
        m0.col0.y * v.x + m0.col1.y * v.y + m0.col2.y * v.z + m0.col3.y * v.w,
        m0.col0.z * v.x + m0.col1.z * v.y + m0.col2.z * v.z + m0.col3.z * v.w,
        m0.col0.w * v.x + m0.col1.w * v.y + m0.col2.w * v.z + m0.col3.w * v.w);
}

inline __host__ __device__ float3 operator*(mat4 &m0, float3 v) {
    float4 v4 = make_float4(
        m0.col0.x * v.x + m0.col1.x * v.y + m0.col2.x * v.z + m0.col3.x * 1,
        m0.col0.y * v.x + m0.col1.y * v.y + m0.col2.y * v.z + m0.col3.y * 1,
        m0.col0.z * v.x + m0.col1.z * v.y + m0.col2.z * v.z + m0.col3.z * 1,
        m0.col0.w * v.x + m0.col1.w * v.y + m0.col2.w * v.z + m0.col3.w * 1);
    return make_float3(v4.x / v4.w, v4.y / v4.w, v4.z / v4.w);
}

//__constant__ Rectangle rect[6];

inline __device__ float scalar_triple(float3 u, float3 v, float3 w) {
    return dot(cross(u, v), w);
}

inline __device__ float ray_intersect_triangle(float3 ray_start, float3 ray_dir, float3 p0, float3 p1, float3 p2, float3* out) {
    float3 pq = ray_dir*INF;
    float3 pa = p0 - ray_start;
    float3 pb = p1 - ray_start;
    float3 pc = p2 - ray_start;

    float u = scalar_triple(pq, pc, pb);
    float v = scalar_triple(pq, pa, pc);
    float w = scalar_triple(pq, pb, pa);

    if ((u <= 0.0 && v <= 0.0 && w <= 0.0) || (u >= 0.0 && v >= 0.0 && w >= 0.0)) {
        float denom = 1.0f / (u + v + w);
        u *= denom;
        v *= denom;
        w *= denom;
        *out = p0*u + p1*v + p2*w;
        return length(*out - ray_start);
    }
    else {
        return -INF;
    }
}

inline __device__ float ray_intersect_rectangle(float3 ray_start, float3 ray_dir, float3 p0, float3 p1, float3 p2, float3 p3, float3* out) {
    float dis = ray_intersect_triangle(ray_start, ray_dir, p0, p1, p2, out);
    if (dis > -INF) {
        return dis;
    }
    else {
        return ray_intersect_triangle(ray_start, ray_dir, p0, p2, p3, out);
    }
}

#endif