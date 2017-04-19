#ifndef MED_IMAGING_ARITHMETRIC_VECTOR2F_H_
#define MED_IMAGING_ARITHMETRIC_VECTOR2F_H_

#include "MedImgArithmetic/mi_arithmetic_common.h"

MED_IMAGING_BEGIN_NAMESPACE

class Point2f;

class Arithmetic_Export Vector2f
{
public:
    union
    {
        __m128 m_Vec128;
        struct
        {
            float x, y;
        }m_Value;
        float _m[2];
    };

public:
    Vector2f()
    {
        m_Vec128 = _mm_setzero_ps();
    }

    inline Vector2f(const Vector2f& vec)
    {
        m_Vec128 = vec.m_Vec128;
    }

    inline Vector2f( const float x, const float y )
    {
        m_Vec128 = _mm_setr_ps(x, y, 0.0f, 0.0f);
    }

    inline Vector2f( const float scalar )
    {
        m_Vec128 = _mm_set1_ps(scalar);
    }

    inline Vector2f( const __m128 vf4 )
    {
        m_Vec128 = vf4;
    }

    inline Vector2f & operator =( const Vector2f &vec )
    {
        m_Vec128 = vec.m_Vec128;
        return *this;
    }

    inline Vector2f & SetX( const float x )
    {
        _vmathVfSetElement(m_Vec128, x, 0);
        return *this;
    }

    inline Vector2f & SetY( const float y )
    {
        _vmathVfSetElement(m_Vec128, y, 1);
        return *this;
    }

    inline const float GetX( ) const
    {
        return _vmathVfGetElement(m_Vec128, 0);
    }

    inline const float GetY( ) const
    {
        return _vmathVfGetElement(m_Vec128, 1);
    }

    inline Vector2f & SetElem( const int idx, const float value )
    {
        _vmathVfSetElement(m_Vec128, value, idx);
        return *this;
    }

    inline const float GetElem( const int idx ) const
    {
        return _vmathVfGetElement(m_Vec128, idx);
    }

    inline const float operator []( const int idx ) const
    {
        return _vmathVfGetElement(m_Vec128, idx);
    }

    inline const __m128 Get128( ) const
    {
        return m_Vec128;
    }

    inline Vector2f & Set128( __m128 vf4)
    {
        m_Vec128 = vf4;
        return *this;
    }

    inline const Vector2f operator +( const Vector2f &vec ) const
    {
        return Vector2f(_mm_add_ps(m_Vec128, vec.m_Vec128));
    }

    inline Vector2f & operator +=( const Vector2f &vec )
    {
        m_Vec128 = _mm_add_ps(m_Vec128, vec.m_Vec128);
        return *this;
    }

    inline const Vector2f operator -( const Vector2f &vec ) const
    {
        return Vector2f(_mm_sub_ps(m_Vec128, vec.m_Vec128));
    }

    inline Vector2f & operator -=( const Vector2f &vec )
    {
        m_Vec128 = _mm_sub_ps(m_Vec128, vec.m_Vec128);
        return *this;
    }

    inline const Vector2f operator -( ) const
    {
        return Vector2f(_mm_sub_ps(_mm_setzero_ps(), m_Vec128));
    }

    inline const Vector2f operator *( const float scalar ) const
    {
        return Vector2f(_mm_mul_ps(m_Vec128, _mm_set1_ps(scalar)));
    }

    inline Vector2f & operator *=( const float scalar )
    {
        m_Vec128 = _mm_mul_ps(m_Vec128, _mm_set1_ps(scalar));
        return *this;
    }

    inline const Vector2f operator /( const float scalar ) const
    {
        return Vector2f(_mm_div_ps(m_Vec128, _mm_set1_ps(scalar)));
    }

    inline Vector2f & operator /=( const float scalar )
    {
        m_Vec128 = _mm_div_ps(m_Vec128, _mm_set1_ps(scalar));
        return *this;
    }

    inline bool operator != ( const Vector2f& vec) const
    {
        __m128 t = fabsf4(_mm_sub_ps(m_Vec128, vec.m_Vec128));
        t = _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1));
        return _vmathVfGetElement(t , 0) > FLOAT_EPSILON;
    }

    inline bool operator == ( const Vector2f& vec) const
    {
        __m128 t = fabsf4(_mm_sub_ps(m_Vec128, vec.m_Vec128));
        t = _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1));
        return _vmathVfGetElement(t , 0) <= FLOAT_EPSILON;
    }

    inline const float MaxElem() const
    {
        __m128 t = _mm_max_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline const float MinElem() const
    {
        __m128 t = _mm_min_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline const float Sum() const
    {
        __m128 t = _mm_add_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline float AngleBetween( const Vector2f& vec) const
    {
        float lenth = this->Magnitude() * vec.Magnitude();
        lenth = (lenth > FLOAT_EPSILON) ? lenth : FLOAT_EPSILON;
        float product = this->DotProduct(vec) / lenth;
        return acos(product);
    }

    inline float DotProduct(const Vector2f& vec) const
    {
        __m128 t = _vmathVfDot2( m_Vec128, vec.m_Vec128 );
        return _vmathVfGetElement(t, 0);
    }

    inline float Magnitude() const
    {
        __m128 t = _mm_sqrt_ps(_vmathVfDot2( m_Vec128, m_Vec128 ));
        return _vmathVfGetElement(t, 0);
    }

    inline void Normalize()
    {
        m_Vec128 = _mm_mul_ps( m_Vec128, newtonrapson_rsqrt4( _vmathVfDot2( m_Vec128, m_Vec128 ) ) );
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Multiply two 2-D vectors per element
Arithmetic_Export  
    const Vector2f MulPerElem( const Vector2f &vec0, const Vector2f &vec1 );

// Divide two 2-D vectors per element
// Floating-point behavior matches standard library function divf4.
Arithmetic_Export  
    const Vector2f DivPerElem( const Vector2f &vec0, const Vector2f &vec1 );

// Compute the reciprocal of a 2-D vector per element
// Floating-point behavior matches standard library function recipf4.
Arithmetic_Export  
    const Vector2f RecipPerElem( const Vector2f &vec );

// Compute the absolute value of a 2-D vector per element
Arithmetic_Export  
    const Vector2f AbsPerElem( const Vector2f &vec );

// Maximum of two 2-D vectors per element
Arithmetic_Export  
    const Vector2f MaxPerElem( const Vector2f &vec0, const Vector2f &vec1 );

// Minimum of two 2-D vectors per element
Arithmetic_Export  
    const Vector2f MinPerElem( const Vector2f &vec0, const Vector2f &vec1 );

// Maximum element of a 2-D vector
Arithmetic_Export  
    const float MaxElem( const Vector2f &vec );

// Minimum element of a 2-D vector
Arithmetic_Export  
    const float MinElem( const Vector2f &vec );

// Compute the sum of all elements of a 2-D vector
Arithmetic_Export  
    const float Sum( const Vector2f &vec );

// Compute the dot product of two 2-D vectors
Arithmetic_Export  
    const float DotProduct( const Vector2f &vec0, const Vector2f &vec1 );

// Compute the square of the length of a 2-D vector
Arithmetic_Export  
    const float LengthSqr( const Vector2f &vec );

// Compute the length of a 2-D vector
Arithmetic_Export  
    const float Length( const Vector2f &vec );

// Normalize a 2-D vector, result is not accurate enough
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export  
    const Vector2f NormalizeApprox( const Vector2f &vec );

// Normalize a 2-D vector
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export  
    const Vector2f Normalize( const Vector2f &vec );

// Linear interpolation between two 3-D vectors
// vec0 * (1 - t) + vec1 * t
Arithmetic_Export  
    const Vector2f Lerp( const float t, const Vector2f &vec0, const Vector2f &vec1 );

// Spherical linear interpolation between two 3-D vectors
// The result is unpredictable if the vectors point in opposite directions.
// Angle = acosf(dot( unitVec0, unitVec1 ));
// recipSinAngle = ( 1.0f / sinf( angle ) );
// scale0 = ( sinf( ( ( 1.0f - t ) * angle ) ) * recipSinAngle );
// scale1 = ( sinf( ( t * angle ) ) * recipSinAngle );
// return ( ( unitVec0 * scale0 ) + ( unitVec1 * scale1 ) );
Arithmetic_Export  
    const Vector2f Slerp( const float t, const Vector2f &unitVec0, const Vector2f &unitVec1 );


// Store x and y elements of 2-D vector in first three words of a float ptr
Arithmetic_Export  
    void StoreXY( const Vector2f &vec, float * fptr );

// Load x and y elements of 2-D vector in first three words of a float ptr
Arithmetic_Export     
    void LoadXY( Vector2f &vec, const float * fptr );


#ifdef _DEBUG
// Print a 2-D vector
// Function is only defined when _DEBUG is defined.
Arithmetic_Export  
    void print( const Vector2f &vec );

// Print a 2-D vector and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export  
    void print( const Vector2f &vec, const char * name );

#endif

MED_IMAGING_END_NAMESPACE

#endif