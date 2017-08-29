#include "mi_vector3f.h"

MED_IMG_BEGIN_NAMESPACE

const Vector3f mul_per_elem(const Vector3f& vec0, const Vector3f& vec1) {
    return Vector3f(_mm_mul_ps(vec0._m128, vec1._m128));
}

const Vector3f div_per_elem(const Vector3f& vec0, const Vector3f& vec1) {
    return Vector3f(_mm_div_ps(vec0._m128, vec1._m128));
}

const Vector3f recip_per_elem(const Vector3f& vec) {
    return Vector3f(_mm_rcp_ps(vec._m128));
}

const Vector3f abs_per_elem(const Vector3f& vec) {
    return Vector3f(fabsf4(vec._m128));
}

const Vector3f max_per_elem(const Vector3f& vec0, const Vector3f& vec1) {
    return Vector3f(_mm_max_ps(vec0._m128, vec1._m128));
}

const Vector3f min_per_elem(const Vector3f& vec0, const Vector3f& vec1) {
    return Vector3f(_mm_min_ps(vec0._m128, vec1._m128));
}

const float max_elem(const Vector3f& vec) {
    __m128 t =
        _mm_max_ps(_mm_max_ps(vec_splat(vec._m128, 0), vec_splat(vec._m128, 1)),
                   vec_splat(vec._m128, 2));
    return _vmathVfGetElement(t, 0);
}

const float min_elem(const Vector3f& vec) {
    __m128 t =
        _mm_min_ps(_mm_min_ps(vec_splat(vec._m128, 0), vec_splat(vec._m128, 1)),
                   vec_splat(vec._m128, 2));
    return _vmathVfGetElement(t, 0);
}

const float sum(const Vector3f& vec) {
    __m128 t =
        _mm_add_ps(_mm_add_ps(vec_splat(vec._m128, 0), vec_splat(vec._m128, 1)),
                   vec_splat(vec._m128, 2));
    return _vmathVfGetElement(t, 0);
}

const float dot_product(const Vector3f& vec0, const Vector3f& vec1) {
    __m128 t = _vmathVfDot3(vec0._m128, vec1._m128);
    return _vmathVfGetElement(t, 0);
}

const float length_sqr(const Vector3f& vec) {
    __m128 t = _vmathVfDot3(vec._m128, vec._m128);
    return _vmathVfGetElement(t, 0);
}

const float length(const Vector3f& vec) {
    __m128 t = _mm_sqrt_ps(_vmathVfDot3(vec._m128, vec._m128));
    return _vmathVfGetElement(t, 0);
}

const Vector3f normalize_approx(const Vector3f& vec) {
    return Vector3f(
               _mm_mul_ps(vec._m128, _mm_rsqrt_ps(_vmathVfDot3(vec._m128, vec._m128))));
}

const Vector3f normalize(const Vector3f& vec) {
    return Vector3f(_mm_mul_ps(
                        vec._m128, newtonrapson_rsqrt4(_vmathVfDot3(vec._m128, vec._m128))));
}

const Vector3f cross(const Vector3f& vec0, const Vector3f& vec1) {
    return Vector3f(_vmathVfCross(vec0._m128, vec1._m128));
}

const Vector3f lerp(const float t, const Vector3f& vec0, const Vector3f& vec1) {
    return (vec0 + ((vec1 - vec0) * t));
}

const Vector3f slerp(const float t, const Vector3f& unitVec0,
                     const Vector3f& unitVec1) {
    const float MY_3D_SLERP_TOL = 0.999f;
    __m128 cosAngle = _vmathVfDot3(unitVec0._m128, unitVec1._m128);
    __m128 selectMask = _mm_cmpgt_ps(_mm_set1_ps(MY_3D_SLERP_TOL), cosAngle);
    __m128 angle = acosf4(cosAngle);
    __m128 tttt = _mm_set1_ps(t);
    __m128 oneMinusT = _mm_sub_ps(_mm_set1_ps(1.0f), tttt);
    __m128 angles = _mm_unpacklo_ps(_mm_set1_ps(1.0f), tttt); // angles = 1, t, 1, t
    angles = _mm_unpacklo_ps(angles, oneMinusT);              // angles = 1, 1-t, t, 1-t
    angles = _mm_mul_ps(angles, angle);
    __m128 sines = sinf4(angles);
    __m128 scales = _mm_div_ps(sines, vec_splat(sines, 0));
    __m128 scale0 = vec_sel(oneMinusT, vec_splat(scales, 1), selectMask);
    __m128 scale1 = vec_sel(tttt, vec_splat(scales, 2), selectMask);
    return Vector3f(
               vec_madd(unitVec0._m128, scale0, _mm_mul_ps(unitVec1._m128, scale1)));
}

void store_xyz(const Vector3f& vec, float* fptr) {
    fptr[0] = vec._value.x;
    fptr[1] = vec._value.y;
    fptr[2] = vec._value.z;
}

void load_xyz(Vector3f& vec, float* fptr) {
    vec = Vector3f(fptr[0], fptr[1], fptr[2]);
}

#ifdef _DEBUG

void print(const Vector3f& vec) {
    union {
        __m128 v;
        float s[4];
    } tmp;
    tmp.v = vec._m128;
    printf("( %f %f %f )\n", tmp.s[0], tmp.s[1], tmp.s[2]);
}

void print(const Vector3f& vec, const char* name) {
    union {
        __m128 v;
        float s[4];
    } tmp;
    tmp.v = vec._m128;
    printf("%s: ( %f %f %f )\n", name, tmp.s[0], tmp.s[1], tmp.s[2]);
}

const Vector3f operator/(const float scalar, const Vector3f& vec) {
    return Vector3f(_mm_div_ps(_mm_set1_ps(scalar), vec._m128));
}

#endif

MED_IMG_END_NAMESPACE
