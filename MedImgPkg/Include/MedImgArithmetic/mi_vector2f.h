#ifndef MED_IMG_ARITHMETRIC_VECTOR2F_H_
#define MED_IMG_ARITHMETRIC_VECTOR2F_H_

#include "MedImgArithmetic/mi_simd.h"

MED_IMG_BEGIN_NAMESPACE

class Point2f;

class Arithmetic_Export Vector2f
{
public:
    union
    {
        __m128 _m128;
        struct
        {
            float x, y;
        }_value;
        float _m[2];
    };

public:
    Vector2f()
    {
        _m128 = _mm_setzero_ps();
    }

    inline Vector2f(const Vector2f& vec)
    {
        _m128 = vec._m128;
    }

    inline Vector2f( const float x, const float y )
    {
        _m128 = _mm_setr_ps(x, y, 0.0f, 0.0f);
    }

    inline Vector2f( const float scalar )
    {
        _m128 = _mm_set1_ps(scalar);
    }

    inline Vector2f( const __m128 vf4 )
    {
        _m128 = vf4;
    }

    inline Vector2f & operator =( const Vector2f &vec )
    {
        _m128 = vec._m128;
        return *this;
    }

    inline Vector2f & set_x( const float x )
    {
        _vmathVfSetElement(_m128, x, 0);
        return *this;
    }

    inline Vector2f & set_y( const float y )
    {
        _vmathVfSetElement(_m128, y, 1);
        return *this;
    }

    inline const float get_x( ) const
    {
        return _vmathVfGetElement(_m128, 0);
    }

    inline const float get_y( ) const
    {
        return _vmathVfGetElement(_m128, 1);
    }

    inline Vector2f & set_elem( const int idx, const float value )
    {
        _vmathVfSetElement(_m128, value, idx);
        return *this;
    }

    inline const float get_elem( const int idx ) const
    {
        return _vmathVfGetElement(_m128, idx);
    }

    inline const float operator []( const int idx ) const
    {
        return _vmathVfGetElement(_m128, idx);
    }

    inline const __m128 get_128( ) const
    {
        return _m128;
    }

    inline Vector2f & set_128( __m128 vf4)
    {
        _m128 = vf4;
        return *this;
    }

    inline const Vector2f operator +( const Vector2f &vec ) const
    {
        return Vector2f(_mm_add_ps(_m128, vec._m128));
    }

    inline Vector2f & operator +=( const Vector2f &vec )
    {
        _m128 = _mm_add_ps(_m128, vec._m128);
        return *this;
    }

    inline const Vector2f operator -( const Vector2f &vec ) const
    {
        return Vector2f(_mm_sub_ps(_m128, vec._m128));
    }

    inline Vector2f & operator -=( const Vector2f &vec )
    {
        _m128 = _mm_sub_ps(_m128, vec._m128);
        return *this;
    }

    inline const Vector2f operator -( ) const
    {
        return Vector2f(_mm_sub_ps(_mm_setzero_ps(), _m128));
    }

    inline const Vector2f operator *( const float scalar ) const
    {
        return Vector2f(_mm_mul_ps(_m128, _mm_set1_ps(scalar)));
    }

    inline Vector2f & operator *=( const float scalar )
    {
        _m128 = _mm_mul_ps(_m128, _mm_set1_ps(scalar));
        return *this;
    }

    inline const Vector2f operator /( const float scalar ) const
    {
        return Vector2f(_mm_div_ps(_m128, _mm_set1_ps(scalar)));
    }

    inline Vector2f & operator /=( const float scalar )
    {
        _m128 = _mm_div_ps(_m128, _mm_set1_ps(scalar));
        return *this;
    }

    inline bool operator != ( const Vector2f& vec) const
    {
        __m128 t = fabsf4(_mm_sub_ps(_m128, vec._m128));
        t = _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1));
        return _vmathVfGetElement(t , 0) > FLOAT_EPSILON;
    }

    inline bool operator == ( const Vector2f& vec) const
    {
        __m128 t = fabsf4(_mm_sub_ps(_m128, vec._m128));
        t = _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1));
        return _vmathVfGetElement(t , 0) <= FLOAT_EPSILON;
    }

    inline const float max_elem() const
    {
        __m128 t = _mm_max_ps( vec_splat( _m128, 0 ), vec_splat( _m128, 1 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline const float min_elem() const
    {
        __m128 t = _mm_min_ps( vec_splat( _m128, 0 ), vec_splat( _m128, 1 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline const float sum() const
    {
        __m128 t = _mm_add_ps( vec_splat( _m128, 0 ), vec_splat( _m128, 1 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline float angle_between( const Vector2f& vec) const
    {
        float lenth = this->magnitude() * vec.magnitude();
        lenth = (lenth > FLOAT_EPSILON) ? lenth : FLOAT_EPSILON;
        float product = this->dot_product(vec) / lenth;
        return acos(product);
    }

    inline float dot_product(const Vector2f& vec) const
    {
        __m128 t = _vmathVfDot2( _m128, vec._m128 );
        return _vmathVfGetElement(t, 0);
    }

    inline float magnitude() const
    {
        __m128 t = _mm_sqrt_ps(_vmathVfDot2( _m128, _m128 ));
        return _vmathVfGetElement(t, 0);
    }

    inline void normalize()
    {
        _m128 = _mm_mul_ps( _m128, newtonrapson_rsqrt4( _vmathVfDot2( _m128, _m128 ) ) );
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Multiply two 2-D vectors per element
Arithmetic_Export  
    const Vector2f mul_per_elem( const Vector2f &vec0, const Vector2f &vec1 );

// Divide two 2-D vectors per element
// Floating-point behavior matches standard library function divf4.
Arithmetic_Export  
    const Vector2f div_per_elem( const Vector2f &vec0, const Vector2f &vec1 );

// Compute the reciprocal of a 2-D vector per element
// Floating-point behavior matches standard library function recipf4.
Arithmetic_Export  
    const Vector2f recip_per_elem( const Vector2f &vec );

// Compute the absolute value of a 2-D vector per element
Arithmetic_Export  
    const Vector2f abs_per_elem( const Vector2f &vec );

// Maximum of two 2-D vectors per element
Arithmetic_Export  
    const Vector2f max_per_elem( const Vector2f &vec0, const Vector2f &vec1 );

// Minimum of two 2-D vectors per element
Arithmetic_Export  
    const Vector2f min_per_elem( const Vector2f &vec0, const Vector2f &vec1 );

// Maximum element of a 2-D vector
Arithmetic_Export  
    const float max_elem( const Vector2f &vec );

// Minimum element of a 2-D vector
Arithmetic_Export  
    const float min_elem( const Vector2f &vec );

// Compute the sum of all elements of a 2-D vector
Arithmetic_Export  
    const float sum( const Vector2f &vec );

// Compute the dot product of two 2-D vectors
Arithmetic_Export  
    const float dot_product( const Vector2f &vec0, const Vector2f &vec1 );

// Compute the square of the length of a 2-D vector
Arithmetic_Export  
    const float length_sqr( const Vector2f &vec );

// Compute the length of a 2-D vector
Arithmetic_Export  
    const float length( const Vector2f &vec );

// normalize a 2-D vector, result is not accurate enough
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export  
    const Vector2f normalize_approx( const Vector2f &vec );

// normalize a 2-D vector
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export  
    const Vector2f normalize( const Vector2f &vec );

// Linear interpolation between two 3-D vectors
// vec0 * (1 - t) + vec1 * t
Arithmetic_Export  
    const Vector2f lerp( const float t, const Vector2f &vec0, const Vector2f &vec1 );

// Spherical linear interpolation between two 3-D vectors
// The result is unpredictable if the vectors point in opposite directions.
// Angle = acosf(dot( unitVec0, unitVec1 ));
// recipSinAngle = ( 1.0f / sinf( angle ) );
// scale0 = ( sinf( ( ( 1.0f - t ) * angle ) ) * recipSinAngle );
// scale1 = ( sinf( ( t * angle ) ) * recipSinAngle );
// return ( ( unitVec0 * scale0 ) + ( unitVec1 * scale1 ) );
Arithmetic_Export  
    const Vector2f slerp( const float t, const Vector2f &unitVec0, const Vector2f &unitVec1 );


// Store x and y elements of 2-D vector in first three words of a float ptr
Arithmetic_Export  
    void store_xy( const Vector2f &vec, float * fptr );

// load x and y elements of 2-D vector in first three words of a float ptr
Arithmetic_Export     
    void load_xy( Vector2f &vec, const float * fptr );


#ifdef _DEBUG
// print a 2-D vector
// Function is only defined when _DEBUG is defined.
Arithmetic_Export  
    void print( const Vector2f &vec );

// print a 2-D vector and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export  
    void print( const Vector2f &vec, const char * name );

#endif

MED_IMG_END_NAMESPACE

#endif