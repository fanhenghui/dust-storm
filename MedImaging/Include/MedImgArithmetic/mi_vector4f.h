#ifndef MED_IMAGING_ARITHMETRIC_VECTOR4F_H_
#define MED_IMAGING_ARITHMETRIC_VECTOR4F_H_

#include "MedImgArithmetic/mi_arithmetic_common.h"
#include "MedImgArithmetic/mi_vector3f.h"

MED_IMAGING_BEGIN_NAMESPACE

/// \class Vector4f
/// 
/// \brief *****
class Arithmetic_Export Vector4f
{
public:
    union
    {
        __m128 m_Vec128;
        struct 
        {
            float x, y, z, w;
        }m_Value;
        float _m[4];
    };

public:
    Vector4f::Vector4f()
    {
        m_Vec128 = _mm_setzero_ps();
    }

    Vector4f::Vector4f( const float x, const float y, const float z, const float w )
    {
        m_Vec128 = _mm_setr_ps(x, y, z, w); 
    }

    Vector4f::Vector4f( const Vector3f &xyz, const float w )
    {
        m_Vec128 = xyz.m_Vec128;
        _vmathVfSetElement(m_Vec128, w, 3);
    }

    Vector4f::Vector4f( const Vector3f &vec )
    {
        m_Vec128 = vec.m_Vec128;
        _vmathVfSetElement(m_Vec128, 0.0f, 3);
    }

    Vector4f::Vector4f( const float scalar )
    {
        m_Vec128 = _mm_set1_ps(scalar);
    }

    Vector4f::Vector4f( const __m128 vf4 )
    {
        m_Vec128 = vf4;
    }

    inline const __m128 Vector4f::Get128() const
    {
        return m_Vec128;
    }

    inline Vector4f & Vector4f::Set128( __m128 vf4 )
    {
        m_Vec128 = vf4;
        return *this;
    }

    inline Vector4f & Vector4f::operator =( const Vector4f &vec )
    {
        m_Vec128 = vec.m_Vec128;
        return *this;
    }

    inline Vector4f & Vector4f::SetXYZ( const Vector3f &vec )
    {
        MCSF_3D_ALIGN16 unsigned int sw[4] = {0, 0, 0, 0xffffffff};
        m_Vec128 = vec_sel( vec.m_Vec128, m_Vec128, sw );
        return *this;
    }

    inline const Vector3f Vector4f::GetXYZ() const
    {
        return Vector3f(m_Vec128);
    }

    inline Vector4f & Vector4f::SetX( const float x )
    {
        _vmathVfSetElement(m_Vec128, x, 0);
        return *this;
    }

    inline Vector4f & Vector4f::SetY( const float y )
    {
        _vmathVfSetElement(m_Vec128, y, 1);
        return *this;
    }

    inline Vector4f & Vector4f::SetZ( const float z )
    {
        _vmathVfSetElement(m_Vec128, z, 2);
        return *this;
    }

    inline Vector4f & Vector4f::SetW( const float w )
    {
        _vmathVfSetElement(m_Vec128, w, 3);
        return *this;
    }

    inline const float Vector4f::GetX() const
    {
        return _vmathVfGetElement(m_Vec128, 0);
        //return m_Value.x;
    }

    inline const float Vector4f::GetY() const
    {
        return _vmathVfGetElement(m_Vec128, 1);
        // return m_Value.y;
    }

    inline const float Vector4f::GetZ() const
    {
        return _vmathVfGetElement(m_Vec128, 2);
        // return m_Value.z;
    }

    inline const float Vector4f::GetW() const
    {
        return _vmathVfGetElement(m_Vec128, 3);
        // return m_Value.w;
    }

    inline Vector4f & Vector4f::SetElem( const int idx, const float value )
    {
        _vmathVfSetElement(m_Vec128, value, idx);
        return *this;
    }

    inline const float Vector4f::GetElem( const int idx ) const
    {
        return _vmathVfGetElement(m_Vec128, idx);
    }

    inline const float Vector4f::operator[]( const int idx ) const
    {
        return _vmathVfGetElement(m_Vec128, idx);
    }

    inline const Vector4f Vector4f::operator+( const Vector4f &vec ) const
    {
        return Vector4f( _mm_add_ps( m_Vec128, vec.m_Vec128 ) );
    }

    inline const Vector4f Vector4f::operator-( const Vector4f &vec ) const
    {
        return Vector4f( _mm_sub_ps( m_Vec128, vec.m_Vec128 ) );
    }

    inline const Vector4f Vector4f::operator*( const float scalar ) const
    {
        return Vector4f( _mm_mul_ps( m_Vec128, _mm_set1_ps(scalar) ) );
    }

    inline const Vector4f Vector4f::operator/( const float scalar ) const
    {
        return Vector4f( _mm_div_ps( m_Vec128, _mm_set1_ps(scalar) ) );
    }

    inline Vector4f & Vector4f::operator+=( const Vector4f &vec )
    {
        m_Vec128 = _mm_add_ps( m_Vec128, vec.m_Vec128 );
        return *this;
    }

    inline Vector4f & Vector4f::operator-=( const Vector4f &vec )
    {
        m_Vec128 = _mm_sub_ps( m_Vec128, vec.m_Vec128 );
        return *this;
    }

    inline Vector4f & Vector4f::operator*=( const float scalar )
    {
        m_Vec128 = _mm_mul_ps( m_Vec128, _mm_set1_ps(scalar) );
        return *this;
    }

    inline Vector4f & Vector4f::operator/=( const float scalar )
    {
        m_Vec128 = _mm_div_ps( m_Vec128, _mm_set1_ps(scalar) );
        return *this;
    }

    inline const Vector4f Vector4f::operator-() const
    {
        return Vector4f(_mm_sub_ps( _mm_setzero_ps(), m_Vec128 ) );
    }

    inline bool Vector4f::operator!=( const Vector4f& vec ) const
    {
        __m128 t = fabsf4(_mm_sub_ps(m_Vec128, vec.m_Vec128));
        t = _mm_add_ps( _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)), _mm_add_ps(vec_splat(t, 2), vec_splat(t, 3)) );
        return _vmathVfGetElement(t , 0) > FLOAT_EPSILON;
    }

    inline bool Vector4f::operator==( const Vector4f& vec ) const
    {
        __m128 t = fabsf4(_mm_sub_ps(m_Vec128, vec.m_Vec128));
        t = _mm_add_ps( _mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)), _mm_add_ps(vec_splat(t, 2), vec_splat(t, 3)) );
        return _vmathVfGetElement(t , 0) <= FLOAT_EPSILON;
    }

    inline const float Vector4f::MaxElem() const
    {
        __m128 t = _mm_max_ps(
            _mm_max_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) ),
            _mm_max_ps( vec_splat( m_Vec128, 2 ), vec_splat( m_Vec128, 3 ) ) );
        return _vmathVfGetElement(t , 0); 
    }

    inline const float Vector4f::MinElem() const
    {
        __m128 t = _mm_min_ps(
            _mm_min_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) ),
            _mm_min_ps( vec_splat( m_Vec128, 2 ), vec_splat( m_Vec128, 3 ) ) );
        return _vmathVfGetElement(t , 0); 
    }

    inline const float Vector4f::Sum() const
    {
        __m128 t = _mm_add_ps(
            _mm_add_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) ),
            _mm_add_ps( vec_splat( m_Vec128, 2 ), vec_splat( m_Vec128, 3 ) ) );
        return _vmathVfGetElement(t , 0); 
    }

    inline float DotProduct( const Vector4f& vec) const
    {
        __m128 t = _vmathVfDot4( m_Vec128, vec.m_Vec128 );
        return _vmathVfGetElement(t, 0);
    }

    inline const float Vector4f::Length() const
    {
        __m128 temp = _mm_sqrt_ps(_vmathVfDot4( m_Vec128, m_Vec128 ));
        return _vmathVfGetElement(temp, 0);
    }

    inline const float Vector4f::LengthSqr() const
    {
        __m128 temp = _vmathVfDot4( m_Vec128, m_Vec128 );
        return _vmathVfGetElement(temp, 0);
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Multiply two 4-D vectors per element
Arithmetic_Export 
    const Vector4f MulPerElem( const Vector4f &vec0, const Vector4f &vec1 );

// Divide two 4-D vectors per element
// Floating-point behavior matches standard library function divf4.
Arithmetic_Export 
    const Vector4f DivPerElem( const Vector4f &vec0, const Vector4f &vec1 );

// Compute the reciprocal of a 4-D vector per element
// Floating-point behavior matches standard library function recipf4.
Arithmetic_Export 
    const Vector4f RecipPerElem( const Vector4f &vec );

// Compute the absolute value of a 4-D vector per element
Arithmetic_Export 
    const Vector4f AbsPerElem( const Vector4f &vec );

// Maximum of two 4-D vectors per element
Arithmetic_Export 
    const Vector4f MaxPerElem( const Vector4f &vec0, const Vector4f &vec1 );

// Minimum of two 4-D vectors per element
Arithmetic_Export 
    const Vector4f MinPerElem( const Vector4f &vec0, const Vector4f &vec1 );

// Maximum element of a 4-D vector
Arithmetic_Export 
    const float MaxElem( const Vector4f &vec );

// Minimum element of a 4-D vector
Arithmetic_Export 
    const float MinElem( const Vector4f &vec );

// Compute the sum of all elements of a 4-D vector
Arithmetic_Export 
    const float Sum( const Vector4f &vec );

// Compute the dot product of two 4-D vectors
Arithmetic_Export 
    const float DotProduct( const Vector4f &vec0, const Vector4f &vec1 );

// Compute the square of the length of a 4-D vector
Arithmetic_Export 
    const float LengthSqr( const Vector4f &vec );

// Compute the length of a 4-D vector
Arithmetic_Export 
    const float Length( const Vector4f &vec );

// Normalize a 4-D vector, result is not accurate enough
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export 
    const Vector4f NormalizeApprox( const Vector4f &vec );

// Normalize a 4-D vector
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export 
    const Vector4f Normalize( const Vector4f &vec );

// Linear interpolation between two 4-D vectors (scalar data contained in vector data type)
// vec0 * (1 - t) + vec1 * t
Arithmetic_Export 
    const Vector4f Lerp( const float t, const Vector4f &vec0, const Vector4f &vec1 );

// Spherical linear interpolation between two 4-D vectors
// The result is unpredictable if the vectors point in opposite directions.
// The result is unpredictable if the vectors point in opposite directions.
// Angle = acosf(dot( unitVec0, unitVec1 ));
// recipSinAngle = ( 1.0f / sinf( angle ) );
// scale0 = ( sinf( ( ( 1.0f - t ) * angle ) ) * recipSinAngle );
// scale1 = ( sinf( ( t * angle ) ) * recipSinAngle );
// return ( ( unitVec0 * scale0 ) + ( unitVec1 * scale1 ) );
Arithmetic_Export 
    const Vector4f Slerp( const float t, const Vector4f &unitVec0, const Vector4f &unitVec1 );

// Store x, y, z and w elements of 4-D vector in first four words of a float ptr
Arithmetic_Export 
    void StoreXYZW( const Vector4f &vec, float * fptr );

// Load x, y, z and w elements of 4-D vector in first four words of a float ptr
Arithmetic_Export 
    void LoadXYZW( Vector4f &vec, const float * fptr );

#ifdef _DEBUG

// Print a 4-D vector
// Function is only defined when _DEBUG is defined.
Arithmetic_Export void print( const Vector4f &vec );

// Print a 4-D vector and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export void print( const Vector4f &vec, const char * name );

#endif

MED_IMAGING_END_NAMESPACE

#endif