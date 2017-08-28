#ifndef MED_IMG_ARITHMETRIC_VECTOR4F_H_
#define MED_IMG_ARITHMETRIC_VECTOR4F_H_

#include "arithmetic/mi_simd.h"
#include "arithmetic/mi_vector3f.h"

MED_IMG_BEGIN_NAMESPACE

/// \class Vector4f
/// 
/// \brief *****
class Arithmetic_Export Vector4f
{
public:
    union
    {
        __m128 _m128;
        struct 
        {
            float x, y, z, w;
        }_value;
        float _m[4];
    };

public:
    Vector4f()
    {
        _m128 = _mm_setzero_ps();
    }

    Vector4f( const float x, const float y, const float z, const float w )
    {
        _m128 = _mm_setr_ps(x, y, z, w); 
    }

    Vector4f( const Vector3f &xyz, const float w )
    {
        _m128 = xyz._m128;
        _vmathVfSetElement(_m128, w, 3);
    }

    Vector4f( const Vector3f &vec )
    {
        _m128 = vec._m128;
        _vmathVfSetElement(_m128, 0.0f, 3);
    }

    Vector4f( const float scalar )
    {
        _m128 = _mm_set1_ps(scalar);
    }

    Vector4f( const __m128 vf4 )
    {
        _m128 = vf4;
    }

    inline const __m128 get_128() const
    {
        return _m128;
    }

    inline Vector4f & set_128( __m128 vf4 )
    {
        _m128 = vf4;
        return *this;
    }

    inline Vector4f & operator =( const Vector4f &vec )
    {
        _m128 = vec._m128;
        return *this;
    }

    inline Vector4f & set_xyz( const Vector3f &vec )
    {
        ALIGN16 unsigned int sw[4] = {0, 0, 0, 0xffffffff};
        _m128 = vec_sel( vec._m128, _m128, sw );
        return *this;
    }

    inline const Vector3f get_xyz() const
    {
        return Vector3f(_m128);
    }

    inline Vector4f & set_x( const float x )
    {
        _vmathVfSetElement(_m128, x, 0);
        return *this;
    }

    inline Vector4f & set_y( const float y )
    {
        _vmathVfSetElement(_m128, y, 1);
        return *this;
    }

    inline Vector4f & set_z( const float z )
    {
        _vmathVfSetElement(_m128, z, 2);
        return *this;
    }

    inline Vector4f & set_w( const float w )
    {
        _vmathVfSetElement(_m128, w, 3);
        return *this;
    }

    inline const float get_x() const
    {
        return _vmathVfGetElement(_m128, 0);
        //return _value.x;
    }

    inline const float get_y() const
    {
        return _vmathVfGetElement(_m128, 1);
        // return _value.y;
    }

    inline const float get_z() const
    {
        return _vmathVfGetElement(_m128, 2);
        // return _value.z;
    }

    inline const float get_w() const
    {
        return _vmathVfGetElement(_m128, 3);
        // return _value.w;
    }

    inline Vector4f & set_elem( const int idx, const float value )
    {
        _vmathVfSetElement(_m128, value, idx);
        return *this;
    }

    inline const float get_elem( const int idx ) const
    {
        return _vmathVfGetElement(_m128, idx);
    }

    inline const float operator[]( const int idx ) const
    {
        return _vmathVfGetElement(_m128, idx);
    }

    inline const Vector4f operator+( const Vector4f &vec ) const
    {
        return Vector4f( _mm_add_ps( _m128, vec._m128 ) );
    }

    inline const Vector4f operator-( const Vector4f &vec ) const
    {
        return Vector4f( _mm_sub_ps( _m128, vec._m128 ) );
    }

    inline const Vector4f operator*( const float scalar ) const
    {
        return Vector4f( _mm_mul_ps( _m128, _mm_set1_ps(scalar) ) );
    }

    inline const Vector4f operator/( const float scalar ) const
    {
        return Vector4f( _mm_div_ps( _m128, _mm_set1_ps(scalar) ) );
    }

    inline Vector4f & operator+=( const Vector4f &vec )
    {
        _m128 = _mm_add_ps( _m128, vec._m128 );
        return *this;
    }

    inline Vector4f & operator-=( const Vector4f &vec )
    {
        _m128 = _mm_sub_ps( _m128, vec._m128 );
        return *this;
    }

    inline Vector4f & operator*=( const float scalar )
    {
        _m128 = _mm_mul_ps( _m128, _mm_set1_ps(scalar) );
        return *this;
    }

    inline Vector4f & operator/=( const float scalar )
    {
        _m128 = _mm_div_ps( _m128, _mm_set1_ps(scalar) );
        return *this;
    }

    inline const Vector4f operator-() const
    {
        return Vector4f(_mm_sub_ps( _mm_setzero_ps(), _m128 ) );
    }

    inline bool operator!=( const Vector4f& vec ) const
    {
        __m128 t = fabsf4(_mm_sub_ps(_m128, vec._m128));
        t = _mm_add_ps( _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)), _mm_add_ps(vec_splat(t, 2), vec_splat(t, 3)) );
        return _vmathVfGetElement(t , 0) > FLOAT_EPSILON;
    }

    inline bool operator==( const Vector4f& vec ) const
    {
        __m128 t = fabsf4(_mm_sub_ps(_m128, vec._m128));
        t = _mm_add_ps( _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)), _mm_add_ps(vec_splat(t, 2), vec_splat(t, 3)) );
        return _vmathVfGetElement(t , 0) <= FLOAT_EPSILON;
    }

    inline const float max_elem() const
    {
        __m128 t = _mm_max_ps(
            _mm_max_ps( vec_splat( _m128, 0 ), vec_splat( _m128, 1 ) ),
            _mm_max_ps( vec_splat( _m128, 2 ), vec_splat( _m128, 3 ) ) );
        return _vmathVfGetElement(t , 0); 
    }

    inline const float min_elem() const
    {
        __m128 t = _mm_min_ps(
            _mm_min_ps( vec_splat( _m128, 0 ), vec_splat( _m128, 1 ) ),
            _mm_min_ps( vec_splat( _m128, 2 ), vec_splat( _m128, 3 ) ) );
        return _vmathVfGetElement(t , 0); 
    }

    inline const float sum() const
    {
        __m128 t = _mm_add_ps(
            _mm_add_ps( vec_splat( _m128, 0 ), vec_splat( _m128, 1 ) ),
            _mm_add_ps( vec_splat( _m128, 2 ), vec_splat( _m128, 3 ) ) );
        return _vmathVfGetElement(t , 0); 
    }

    inline float dot_product( const Vector4f& vec) const
    {
        __m128 t = _vmathVfDot4( _m128, vec._m128 );
        return _vmathVfGetElement(t, 0);
    }

    inline const float length() const
    {
        __m128 temp = _mm_sqrt_ps(_vmathVfDot4( _m128, _m128 ));
        return _vmathVfGetElement(temp, 0);
    }

    inline const float length_sqr() const
    {
        __m128 temp = _vmathVfDot4( _m128, _m128 );
        return _vmathVfGetElement(temp, 0);
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Multiply two 4-D vectors per element
Arithmetic_Export 
    const Vector4f mul_per_elem( const Vector4f &vec0, const Vector4f &vec1 );

// Divide two 4-D vectors per element
// Floating-point behavior matches standard library function divf4.
Arithmetic_Export 
    const Vector4f div_per_elem( const Vector4f &vec0, const Vector4f &vec1 );

// Compute the reciprocal of a 4-D vector per element
// Floating-point behavior matches standard library function recipf4.
Arithmetic_Export 
    const Vector4f recip_per_elem( const Vector4f &vec );

// Compute the absolute value of a 4-D vector per element
Arithmetic_Export 
    const Vector4f abs_per_elem( const Vector4f &vec );

// Maximum of two 4-D vectors per element
Arithmetic_Export 
    const Vector4f max_per_elem( const Vector4f &vec0, const Vector4f &vec1 );

// Minimum of two 4-D vectors per element
Arithmetic_Export 
    const Vector4f min_per_elem( const Vector4f &vec0, const Vector4f &vec1 );

// Maximum element of a 4-D vector
Arithmetic_Export 
    const float max_elem( const Vector4f &vec );

// Minimum element of a 4-D vector
Arithmetic_Export 
    const float min_elem( const Vector4f &vec );

// Compute the sum of all elements of a 4-D vector
Arithmetic_Export 
    const float sum( const Vector4f &vec );

// Compute the dot product of two 4-D vectors
Arithmetic_Export 
    const float dot_product( const Vector4f &vec0, const Vector4f &vec1 );

// Compute the square of the length of a 4-D vector
Arithmetic_Export 
    const float length_sqr( const Vector4f &vec );

// Compute the length of a 4-D vector
Arithmetic_Export 
    const float length( const Vector4f &vec );

// normalize a 4-D vector, result is not accurate enough
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export 
    const Vector4f normalize_approx( const Vector4f &vec );

// normalize a 4-D vector
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export 
    const Vector4f normalize( const Vector4f &vec );

// Linear interpolation between two 4-D vectors (scalar data contained in vector data type)
// vec0 * (1 - t) + vec1 * t
Arithmetic_Export 
    const Vector4f lerp( const float t, const Vector4f &vec0, const Vector4f &vec1 );

// Spherical linear interpolation between two 4-D vectors
// The result is unpredictable if the vectors point in opposite directions.
// The result is unpredictable if the vectors point in opposite directions.
// Angle = acosf(dot( unitVec0, unitVec1 ));
// recipSinAngle = ( 1.0f / sinf( angle ) );
// scale0 = ( sinf( ( ( 1.0f - t ) * angle ) ) * recipSinAngle );
// scale1 = ( sinf( ( t * angle ) ) * recipSinAngle );
// return ( ( unitVec0 * scale0 ) + ( unitVec1 * scale1 ) );
Arithmetic_Export 
    const Vector4f slerp( const float t, const Vector4f &unitVec0, const Vector4f &unitVec1 );

// Store x, y, z and w elements of 4-D vector in first four words of a float ptr
Arithmetic_Export 
    void store_xyzw( const Vector4f &vec, float * fptr );

// load x, y, z and w elements of 4-D vector in first four words of a float ptr
Arithmetic_Export 
    void load_xyzw( Vector4f &vec, const float * fptr );

#ifdef _DEBUG

// print a 4-D vector
// Function is only defined when _DEBUG is defined.
Arithmetic_Export void print( const Vector4f &vec );

// print a 4-D vector and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export void print( const Vector4f &vec, const char * name );

#endif

MED_IMG_END_NAMESPACE

#endif