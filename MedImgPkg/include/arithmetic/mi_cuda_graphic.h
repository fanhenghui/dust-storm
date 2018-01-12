#ifndef MED_IMG_ARITHMETIC_MI_CUDA_GRAPHIC_H
#define MED_IMG_ARITHMETIC_MI_CUDA_GRAPHIC_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "arithmetic/mi_cuda_math.h"
#include "arithmetic/mi_matrix4.h"

#define INF     2e10f
#define EPSILON 1e-6f

struct Viewport {
    int x, y, width, height;
    __host__ __device__ Viewport(int x_, int y_, int width_, int height_) : x(x_), y(y_), width(width_), height(height_) {}
};

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

inline __device__ float scalar_triple(float3 u, float3 v, float3 w) {
    return dot(cross(u, v), w);
}

inline __device__ float ray_intersect_triangle(float3 ray_start, float3 ray_dir, float3 p0, float3 p1, float3 p2, float3 *uvw, float3* out) {
    float3 pq = ray_dir*INF;
    float3 pa = p0 - ray_start;
    float3 pb = p1 - ray_start;
    float3 pc = p2 - ray_start;

    uvw->x = scalar_triple(pq, pc, pb);
    uvw->y = scalar_triple(pq, pa, pc);
    uvw->z = scalar_triple(pq, pb, pa);

    float epsilon = 0.0f;
    if ((uvw->x <= epsilon && uvw->y <= epsilon && uvw->z <= epsilon) || (uvw->x >= -epsilon && uvw->y >= -epsilon && uvw->z >= -epsilon)) {
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

inline __device__ bool in_line(float3 p0, float3 p1, float3 p) {
    if (fabs(length(p1 - p0) - length(p0 - p) - length(p1 - p)) < 1e-4f) {
        return true;
    }
    else {
        return false;
    }
}

inline __device__ float ray_intersect_rectangle(float3 ray_start, float3 ray_dir, float3 p0, float3 p1, float3 p2, float3 p3, float3* __restrict__ uvw, float3* __restrict__ out, int& tri_type) {
    float3 pq = ray_dir*INF;
    float3 pp0 = p0 - ray_start;
    float3 pp1 = p1 - ray_start;
    float3 pp2 = p2 - ray_start;
    float3 pp3 = p3 - ray_start;
    float denom;
    float epsilon = 0.0f;

    //triangle p0p1p2
    uvw->x = scalar_triple(pq, pp2, pp1);//21
    uvw->y = scalar_triple(pq, pp0, pp2);//02
    uvw->z = scalar_triple(pq, pp1, pp0);//10
                                         //restrict in triangle(without border)
    if ((uvw->x <= epsilon && uvw->y <= epsilon && uvw->z <= epsilon) || (uvw->x >= -epsilon && uvw->y >= -epsilon && uvw->z >= -epsilon)) {
        tri_type = 0;
        denom = 1.0f / (uvw->x + uvw->y + uvw->z);
        *uvw *= denom;
        *out = p0*uvw->x + p1*uvw->y + p2*uvw->z;
        return length(*out - ray_start);
    }
    else {
        denom = 1.0f / (uvw->x + uvw->y + uvw->z);
        *uvw *= denom;
        *out = p0*uvw->x + p1*uvw->y + p2*uvw->z;
        if (in_line(p0, p2, *out)) {
            tri_type = 0;
            return length(*out - ray_start);
        }
    }

    //triangle p0p2p3
    uvw->x = scalar_triple(pq, pp3, pp2);
    uvw->y = scalar_triple(pq, pp0, pp3);
    uvw->z = scalar_triple(pq, pp2, pp0);
    //restrict in triangle(without border)
    if ((uvw->x <= epsilon && uvw->y <= epsilon && uvw->z <= epsilon) || (uvw->x >= -epsilon && uvw->y >= -epsilon && uvw->z >= -epsilon)) {
        tri_type = 1;
        denom = 1.0f / (uvw->x + uvw->y + uvw->z);
        *uvw *= denom;
        *out = p0*uvw->x + p2*uvw->y + p3*uvw->z;
        return length(*out - ray_start);
    }
    else {
        denom = 1.0f / (uvw->x + uvw->y + uvw->z);
        *uvw *= denom;
        *out = p0*uvw->x + p2*uvw->y + p3*uvw->z;
        if (in_line(p0, p2, *out)) {
            tri_type = 1;
            return length(*out - ray_start);
        }
    }

    return -INF;
}

#endif