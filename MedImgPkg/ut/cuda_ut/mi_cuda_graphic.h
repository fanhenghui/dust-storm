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

struct cudaGLTextureReadOnly {
    GLuint gl_tex_id;
    GLenum target;
    cudaTextureObject_t cuda_tex_obj;//GLTexture2D->cudaTextureType2D
    cudaGraphicsResource *cuda_res;
    cudaArray* d_cuda_array;
};

inline  __host__ void register_image(cudaGLTextureReadOnly& cuda_tex) {
    cudaGraphicsGLRegisterImage(&cuda_tex.cuda_res, cuda_tex.gl_tex_id, cuda_tex.target, cudaGraphicsRegisterFlagsReadOnly);
    CHECK_CUDA_ERROR;
}

inline __host__ void bind_texture(cudaGLTextureReadOnly& cuda_tex, bool normalized_coords) {
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_tex.d_cuda_array;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
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

//struct Triangle {
//    float3 p0, p1, p2;//position coordiante(Model)
//    float4 c0, c1, c2;//vertex color
//    float2 t0, t1, t2;//texture coordinat
//};
//
//__host__ __device__ inline void make_triangle(Triangle& tri, float3 p0, float3 p1, float3 p2) {
//    tri.p0 = p0;
//    tri.p1 = p1;
//    tri.p2 = p2;
//}
//
//__host__ __device__ inline void make_triangle(Triangle& tri, float3 p0, float3 p1, float3 p2, float4 c0, float4 c1, float4 c2) {
//    tri.p0 = p0;
//    tri.p1 = p1;
//    tri.p2 = p2;
//    tri.c0 = c0;
//    tri.c1 = c1;
//    tri.c2 = c2;
//}
//
//__host__ __device__ inline void make_triangle(Triangle& tri, float3 p0, float3 p1, float3 p2, float2 t0, float2 t1, float2 t2) {
//    tri.p0 = p0;
//    tri.p1 = p1;
//    tri.p2 = p2;
//    tri.t0 = t0;
//    tri.t1 = t1;
//    tri.t2 = t2;
//}


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

inline __host__ __device__ float det(mat4 &m) {
    float v0 = m.col0.z * m.col1.w - m.col1.z * m.col0.w;
    float v1 = m.col0.z * m.col2.w - m.col2.z * m.col0.w;
    float v2 = m.col0.z * m.col3.w - m.col3.z * m.col0.w;
    float v3 = m.col1.z * m.col2.w - m.col2.z * m.col1.w;
    float v4 = m.col1.z * m.col3.w - m.col3.z * m.col1.w;
    float v5 = m.col2.z * m.col3.w - m.col3.z * m.col2.w;

    float t00 = +(v5 * m.col1.y - v4 * m.col2.y + v3 * m.col3.y);
    float t10 = -(v5 * m.col0.y - v2 * m.col2.y + v1 * m.col3.y);
    float t20 = +(v4 * m.col0.y - v2 * m.col1.y + v0 * m.col3.y);
    float t30 = -(v3 * m.col0.y - v1 * m.col1.y + v0 * m.col2.y);

    return (t00 * m.col0.x + t10 * m.col1.x + t20 * m.col2.x + t30 * m.col3.x);
};

inline __device__ float scalar_triple(float3 u, float3 v, float3 w) {
    return dot(cross(u, v), w);
}

//calculate d00 d01 d11 denom
inline __device__ bool triangle_barycentric(float3 p0, float3 p1, float3 p2, float3 p, float3& uvw) {
    float3 v0 = p1 - p0;
    float3 v1 = p2 - p0;
    float3 v2 = p - p0;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00*d11 - d01*d01;
    if (fabs(denom) < 1e-5) {
        return false;
    }
    
    float v = (d11*d20 - d01*d21) / denom;
    float w = (d00*d21 - d01*d20) / denom;
    float u = 1.0 - w - v;
    uvw.x = u;
    uvw.y = v;
    uvw.z = w;
    return uvw.x >= 0 && uvw.y >= 0 && uvw.z >= 0;
}

inline __device__ bool triangle_barycentric_2d(float2 p0, float2 p1, float2 p2, float2 p, float3& uvw) {
    float depth = 10;
    float2 v0 = p1 * depth - p0 * depth;
    float2 v1 = p2 * depth - p0 * depth;
    float2 v2 = p * depth - p0 * depth;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00*d11 - d01*d01;
    if (fabs(denom) < 1e-5) {
        return false;
    }

    float v = (d11*d20 - d01*d21) / denom;
    float w = (d00*d21 - d01*d20) / denom;
    float u = 1.0 - w - v;  
    uvw.x = u;
    uvw.y = v;
    uvw.z = w;
    return uvw.x >= 0 && uvw.y >= 0 && uvw.z >= 0;
}

//d00 d11 d11 denom
inline __device__ void triangle_barycentric_cache(float3 p0, float3 p1, float3 p2, float4& cache_param) {
    float3 v0 = p1 - p0;
    float3 v1 = p2 - p0;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float denom = d00*d11 - d01*d01;
    cache_param.x = d00;
    cache_param.y = d01;
    cache_param.z = d11;
    cache_param.w = denom;
}

inline __device__ void triangle_barycentric_cache_2d(float2 p0, float2 p1, float2 p2, float4& cache_param) {
    float2 v0 = p1 - p0;
    float2 v1 = p2 - p0;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float denom = d00*d11 - d01*d01;
    cache_param.x = d00;
    cache_param.y = d01;
    cache_param.z = d11;
    cache_param.w = denom;
}

inline __device__ float3 triangle_barycentric_ext(float3 p0, float3 p1, float3 p2, float3 p, float4 cache_param, float3& uvw) {
    float3 v0 = p1 - p0;
    float3 v1 = p2 - p0;
    float3 v2 = p - p0;
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    uvw.x = (cache_param.z*d20 - cache_param.y*d21) / cache_param.w;
    uvw.y = (cache_param.x*d21 - cache_param.y*d20) / cache_param.w;
    uvw.z = 1.0 - uvw.x - uvw.y;
}

inline __device__ float3 triangle_barycentric_ext_2d(float2 p0, float2 p1, float2 p2, float2 p, float4 cache_param, float3& uvw) {
    float2 v0 = p1 - p0;
    float2 v1 = p2 - p0;
    float2 v2 = p - p0;
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    uvw.x = (cache_param.z*d20 - cache_param.y*d21) / cache_param.w;
    uvw.y = (cache_param.x*d21 - cache_param.y*d20) / cache_param.w;
    uvw.z = 1.0 - uvw.x - uvw.y;
}


inline __device__ float ray_intersect_triangle(float3 ray_start, float3 ray_dir, float3 p0, float3 p1, float3 p2, float3 *uvw, float3* out) {
    float3 pq = ray_dir*INF;
    float3 pa = p0 - ray_start;
    float3 pb = p1 - ray_start;
    float3 pc = p2 - ray_start;

    uvw->x = scalar_triple(pq, pc, pb);
    uvw->y = scalar_triple(pq, pa, pc);
    uvw->z = scalar_triple(pq, pb, pa);

    if ((uvw->x <= 0.0 && uvw->y <= 0.0 && uvw->z <= 0.0) || (uvw->x >= 0.0 && uvw->y >= 0.0 && uvw->z >= 0.0)) {
        float denom = 1.0f / (uvw->x + uvw->y + uvw->z);
        uvw->x *= denom;
        uvw->y *= denom;
        uvw->z *= denom;
        *out = p0*uvw->x + p1*uvw->y + p2*uvw->z;
        return length(*out - ray_start);
    }
    else {
        return -INF;
    }
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