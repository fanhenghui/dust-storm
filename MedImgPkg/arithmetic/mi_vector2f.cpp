#include "mi_vector2f.h"

MED_IMG_BEGIN_NAMESPACE

const Vector2f mul_per_elem(const Vector2f &vec0, const Vector2f &vec1) {
    return Vector2f(_mm_mul_ps(vec0._m128, vec1._m128));
}

const Vector2f div_per_elem(const Vector2f &vec0, const Vector2f &vec1) {
    return Vector2f(_mm_div_ps(vec0._m128, vec1._m128));
}

const Vector2f recip_per_elem(const Vector2f &vec) {
    return Vector2f(_mm_rcp_ps(vec._m128));
}

const Vector2f abs_per_elem(const Vector2f &vec) {
    return Vector2f(fabsf4(vec._m128));
}

const Vector2f max_per_elem(const Vector2f &vec0, const Vector2f &vec1) {
    return Vector2f(_mm_max_ps(vec0._m128, vec1._m128));
}

const Vector2f min_per_elem(const Vector2f &vec0, const Vector2f &vec1) {
    return Vector2f(_mm_min_ps(vec0._m128, vec1._m128));
}

const float max_elem(const Vector2f &vec) {
    __m128 t = _mm_max_ps(vec_splat(vec._m128, 0), vec_splat(vec._m128, 1));
    return _vmathVfGetElement(t, 0);
}

const float min_elem(const Vector2f &vec) {
    __m128 t = _mm_min_ps(vec_splat(vec._m128, 0), vec_splat(vec._m128, 1));
    return _vmathVfGetElement(t, 0);
}

const float sum(const Vector2f &vec) {
    __m128 t = _mm_add_ps(vec_splat(vec._m128, 0), vec_splat(vec._m128, 1));
    return _vmathVfGetElement(t, 0);
}

const float dot_product(const Vector2f &vec0, const Vector2f &vec1) {
    __m128 t = _vmathVfDot2(vec0._m128, vec1._m128);
    return _vmathVfGetElement(t, 0);
}

const float length_sqr(const Vector2f &vec) {
    __m128 t = _vmathVfDot2(vec._m128, vec._m128);
    return _vmathVfGetElement(t, 0);
}

const float length(const Vector2f &vec) {
    __m128 t = _mm_sqrt_ps(_vmathVfDot2(vec._m128, vec._m128));
    return _vmathVfGetElement(t, 0);
}

const Vector2f normalize_approx(const Vector2f &vec) {
    return Vector2f(
               _mm_mul_ps(vec._m128, _mm_rsqrt_ps(_vmathVfDot2(vec._m128, vec._m128))));
}

const Vector2f normalize(const Vector2f &vec) {
    return Vector2f(_mm_mul_ps(
                        vec._m128, newtonrapson_rsqrt4(_vmathVfDot2(vec._m128, vec._m128))));
}

const Vector2f lerp(const float t, const Vector2f &vec0, const Vector2f &vec1) {
    return (vec0 + ((vec1 - vec0) * t));
}

const Vector2f slerp(const float t, const Vector2f &unitVec0,
                     const Vector2f &unitVec1) {
    const float MY_3D_SLERP_TOL = 0.999f;
    __m128 cosAngle = _vmathVfDot2(unitVec0._m128, unitVec1._m128);
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
    return Vector2f(
               vec_madd(unitVec0._m128, scale0, _mm_mul_ps(unitVec1._m128, scale1)));
}

void store_xy(const Vector2f &vec, float *fptr) {
    fptr[0] = vec.get_x();
    fptr[1] = vec.get_y();
}

void load_xy(Vector2f &vec, const float *fptr) {
    vec = Vector2f(fptr[0], fptr[1]);
}

#ifdef _DEBUG

void print(const Vector2f &vec) {
    union {
        __m128 v;
        float s[2];
    } tmp;
    tmp.v = vec._m128;
    printf("( %f %f )\n", tmp.s[0], tmp.s[1]);
}

void print(const Vector2f &vec, const char *name) {
    union {
        __m128 v;
        float s[2];
    } tmp;
    tmp.v = vec._m128;
    printf("%s: ( %f %f )\n", name, tmp.s[0], tmp.s[1]);
}
#endif

MED_IMG_END_NAMESPACE