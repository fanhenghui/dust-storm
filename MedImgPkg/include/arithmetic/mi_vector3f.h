#ifndef MEDIMGARITHMETIC_MI_VECTOR3F_H
#define MEDIMGARITHMETIC_MI_VECTOR3F_H

#include "arithmetic/mi_simd.h"

MED_IMG_BEGIN_NAMESPACE

class Point3f;

class Arithmetic_Export Vector3f {
public:
    union {
        __m128 _m128;
        struct {
            float x, y, z;
        } _value;
        float _m[4];
    };

public:
    Vector3f() {
        _m128 = _mm_setzero_ps();
    }

    Vector3f(const Vector3f &vec) {
        _m128 = vec._m128;
    }

    Vector3f(const float _x, const float _y, const float _z) {
        _m128 = _mm_setr_ps(_x, _y, _z, 0.0f);
    }

    Vector3f(const float scalar) {
        _m128 = _mm_set1_ps(scalar);
    }

    Vector3f(const __m128 vf4) {
        _m128 = vf4;
    }

    inline Vector3f &operator=(const Vector3f &vec) {
        _m128 = vec._m128;
        return *this;
    }

    inline Vector3f &set_x(const float x) {
        _vmathVfSetElement(_m128, x, 0);
        return *this;
    }

    inline Vector3f &set_y(const float x) {
        _vmathVfSetElement(_m128, x, 1);
        return *this;
    }

    inline Vector3f &set_z(const float x) {
        _vmathVfSetElement(_m128, x, 2);
        return *this;
    }

    inline const float get_x() const {
        return _vmathVfGetElement(_m128, 0);
        // return _value.x;
    }

    inline const float get_y() const {
        return _vmathVfGetElement(_m128, 1);
        // return _value.y;
    }

    inline const float get_z() const {
        return _vmathVfGetElement(_m128, 2);
        // return _value.z;
    }

    inline Vector3f &set_elem(const int idx, const float value) {
        _vmathVfSetElement(_m128, value, idx);
        return *this;
    }

    inline const float get_elem(const int idx) const {
        return _vmathVfGetElement(_m128, idx);
    }

    inline const float operator[](const int idx) const {
        return _vmathVfGetElement(_m128, idx);
    }

    inline const __m128 get_128() const {
        return _m128;
    }

    inline Vector3f &set_128(__m128 vec) {
        _m128 = vec;
        return *this;
    }

    inline const Vector3f operator+(const Vector3f &vec) const {
        return Vector3f(_mm_add_ps(_m128, vec._m128));
    }

    inline Vector3f &operator+=(const Vector3f &vec) {
        _m128 = _mm_add_ps(_m128, vec._m128);
        return *this;
    }

    inline const Vector3f operator-(const Vector3f &vec) const {
        return Vector3f(_mm_sub_ps(_m128, vec._m128));
    }

    inline Vector3f &operator-=(const Vector3f &vec) {
        _m128 = _mm_sub_ps(_m128, vec._m128);
        return *this;
    }

    inline const Vector3f operator-() const {
        return Vector3f(_mm_sub_ps(_mm_setzero_ps(), _m128));
    }

    inline const Vector3f operator*(const float scalar) const {
        return Vector3f(_mm_mul_ps(_m128, _mm_set1_ps(scalar)));
    }

    inline const Vector3f operator*(const Vector3f &vec) const {
        return Vector3f(_mm_mul_ps(_m128, vec._m128));
    }

    inline Vector3f &operator*=(const float scalar) {
        _m128 = _mm_mul_ps(_m128, _mm_set1_ps(scalar));
        return *this;
    }

    inline const Vector3f operator/(const float scalar) const {
        return Vector3f(_mm_div_ps(_m128, _mm_set1_ps(scalar)));
    }

    inline const Vector3f operator/(const Vector3f &vec) const {
        return Vector3f(_mm_div_ps(_m128, vec._m128));
    }

    inline Vector3f &operator/=(const float scalar) {
        _m128 = _mm_div_ps(_m128, _mm_set1_ps(scalar));
        return *this;
    }

    inline Vector3f &operator/=(const Vector3f &vec) {
        _m128 = _mm_div_ps(_m128, vec._m128);
        return *this;
    }

    inline bool operator!=(const Vector3f &vec) const {
        __m128 t = fabsf4(_mm_sub_ps(_m128, vec._m128));
        t = _mm_add_ps(_mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)),
                       vec_splat(t, 2));
        return _vmathVfGetElement(t, 0) > FLOAT_EPSILON;
    }

    inline bool operator==(const Vector3f &vec) const {
        __m128 t = fabsf4(_mm_sub_ps(_m128, vec._m128));
        t = _mm_add_ps(_mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)),
                       vec_splat(t, 2));
        return _vmathVfGetElement(t, 0) <= FLOAT_EPSILON;
    }

    inline float angle_between(const Vector3f &vec) const {
        float dlen = this->magnitude() * vec.magnitude();
        dlen = (dlen > FLOAT_EPSILON) ? dlen : FLOAT_EPSILON;
        float dprod = this->dot_product(vec) / dlen;
        return acos(dprod);
    }

    inline Vector3f cross_product(const Vector3f &vec) const {
        return Vector3f(_vmathVfCross(_m128, vec._m128));
    }

    inline float dot_product(const Vector3f &vec) const {
        __m128 t = _vmathVfDot3(_m128, vec._m128);
        return _vmathVfGetElement(t, 0);
    }

    inline float magnitude() const {
        __m128 t = _mm_sqrt_ps(_vmathVfDot3(_m128, _m128));
        return _vmathVfGetElement(t, 0);
    }

    inline void normalize() {
        _m128 = _mm_mul_ps(_m128, newtonrapson_rsqrt4(_vmathVfDot3(_m128, _m128)));
    }

    inline Vector3f get_normalize() const {
        return Vector3f(
                   _mm_mul_ps(_m128, newtonrapson_rsqrt4(_vmathVfDot3(_m128, _m128))));
    }

    inline Vector3f reflect(const Vector3f &normal) const {
        return Vector3f(*this - (normal * 2.0f * this->dot_product(normal)));
    }

    inline const float max_elem() const {
        __m128 t = _mm_max_ps(_mm_max_ps(vec_splat(_m128, 0), vec_splat(_m128, 1)),
                              vec_splat(_m128, 2));
        return _vmathVfGetElement(t, 0);
    }

    inline const float min_elem() const {
        __m128 t = _mm_min_ps(_mm_min_ps(vec_splat(_m128, 0), vec_splat(_m128, 1)),
                              vec_splat(_m128, 2));
        return _vmathVfGetElement(t, 0);
    }

    inline Vector3f max_per_elem(const Vector3f &v) const {
        return Vector3f(_mm_max_ps(_m128, v._m128));
    }

    inline Vector3f min_per_elem(const Vector3f &v) const {
        return Vector3f(_mm_min_ps(_m128, v._m128));
    }

    inline const float sum() const {
        __m128 t = _mm_add_ps(_mm_add_ps(vec_splat(_m128, 0), vec_splat(_m128, 1)),
                              vec_splat(_m128, 2));
        return _vmathVfGetElement(t, 0);
    }

    inline const float length() const {
        __m128 t = _mm_sqrt_ps(_vmathVfDot3(_m128, _m128));
        return _vmathVfGetElement(t, 0);
    }

    inline const float length_sqr() const {
        __m128 t = _vmathVfDot3(_m128, _m128);
        return _vmathVfGetElement(t, 0);
    }

    inline Vector3f less_than(const Vector3f &v) const {
        return Vector3f(_mm_cmplt_ps(_m128, v._m128));
    }

    inline Vector3f to_abs() const {
        return Vector3f(_mm_and_ps(_m128, toM128(0x7fffffff)));
    }

    inline void abs() {
        _m128 = _mm_and_ps(_m128, toM128(0x7fffffff));
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Multiply two 3-D vectors per element
Arithmetic_Export const Vector3f mul_per_elem(const Vector3f &vec0,
        const Vector3f &vec1);

// Divide two 3-D vectors per element
// Floating-point behavior matches standard library function divf4.
Arithmetic_Export const Vector3f div_per_elem(const Vector3f &vec0,
        const Vector3f &vec1);

// Compute the reciprocal of a 3-D vector per element
// Floating-point behavior matches standard library function recipf4.
Arithmetic_Export const Vector3f recip_per_elem(const Vector3f &vec);

// Compute the absolute value of a 3-D vector per element
Arithmetic_Export const Vector3f abs_per_elem(const Vector3f &vec);

// Maximum of two 3-D vectors per element
Arithmetic_Export const Vector3f max_per_elem(const Vector3f &vec0,
        const Vector3f &vec1);

// Minimum of two 3-D vectors per element
Arithmetic_Export const Vector3f min_per_elem(const Vector3f &vec0,
        const Vector3f &vec1);

// Maximum element of a 3-D vector
Arithmetic_Export const float max_elem(const Vector3f &vec);

// Minimum element of a 3-D vector
Arithmetic_Export const float min_elem(const Vector3f &vec);

// Compute the sum of all elements of a 3-D vector
Arithmetic_Export const float sum(const Vector3f &vec);

// Compute the dot product of two 3-D vectors
Arithmetic_Export const float dot_product(const Vector3f &vec0,
        const Vector3f &vec1);

// Compute the square of the length of a 3-D vector
Arithmetic_Export const float length_sqr(const Vector3f &vec);

// Compute the length of a 3-D vector
Arithmetic_Export const float length(const Vector3f &vec);

// normalize a 3-D vector, result is not accurate enough
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export const Vector3f normalize_approx(const Vector3f &vec);

// normalize a 3-D vector, using Newton iteration to refine rsqrt operation
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export const Vector3f normalize(const Vector3f &vec);

// Compute cross product of two 3-D vectors
Arithmetic_Export const Vector3f cross(const Vector3f &vec0,
                                       const Vector3f &vec1);

// Linear interpolation between two 3-D vectors
// vec0 * (1 - t) + vec1 * t
Arithmetic_Export const Vector3f lerp(const float t, const Vector3f &vec0,
                                      const Vector3f &vec1);

// Spherical linear interpolation between two 3-D vectors
// The result is unpredictable if the vectors point in opposite directions.
// Angle = acosf(dot( unitVec0, unitVec1 ));
// recipSinAngle = ( 1.0f / sinf( angle ) );
// scale0 = ( sinf( ( ( 1.0f - t ) * angle ) ) * recipSinAngle );
// scale1 = ( sinf( ( t * angle ) ) * recipSinAngle );
// return ( ( unitVec0 * scale0 ) + ( unitVec1 * scale1 ) );
Arithmetic_Export const Vector3f slerp(const float t, const Vector3f &unitVec0,
                                       const Vector3f &unitVec1);

// Store x, y, and z elements of 3-D vector in first three words of a float ptr,
// preserving fourth word
Arithmetic_Export void store_xyz(const Vector3f &vec, float *fptr);

// load x, y, and z elements of 3-D vector in first three words of a float ptr,
// preserving fourth word
Arithmetic_Export void load_xyz(Vector3f &vec, const float *fptr);

Arithmetic_Export const Vector3f operator/(const float scalar,
        const Vector3f &vec);

#ifdef _DEBUG
// print a 3-D vector
// Function is only defined when _DEBUG is defined.
Arithmetic_Export void print(const Vector3f &vec);

// print a 3-D vector and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export void print(const Vector3f &vec, const char *name);
#endif

MED_IMG_END_NAMESPACE
#endif
