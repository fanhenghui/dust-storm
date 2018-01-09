#ifndef MED_IMG_ARITHMETIC_MI_CUDA_GRAPHIC_H
#define MED_IMG_ARITHMETIC_MI_CUDA_GRAPHIC_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "arithmetic/mi_cuda_math.h"
#include "arithmetic/mi_matrix4.h"

struct mat4 {
    float4 col0, col1, col2, col3;
    __host__ __device__ mat4(float4 col0_, float4 col1_, float4 col2_, float4 col3_) :col0(col0_), col1(col1_), col2(col2_), col3(col3_) {
    }
    __host__ __device__ mat4() {}
};

inline mat4 matrix4_to_mat4(const medical_imaging::Matrix4& mat44) {
    return mat4(
        make_float4((float)mat44._m[0],  (float)mat44._m[1],  (float)mat44._m[2],  (float)mat44._m[3]),
        make_float4((float)mat44._m[4],  (float)mat44._m[5],  (float)mat44._m[6],  (float)mat44._m[7]),
        make_float4((float)mat44._m[8],  (float)mat44._m[9],  (float)mat44._m[10], (float)mat44._m[11]),
        make_float4((float)mat44._m[12], (float)mat44._m[13], (float)mat44._m[14], (float)mat44._m[15]));
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

#endif
