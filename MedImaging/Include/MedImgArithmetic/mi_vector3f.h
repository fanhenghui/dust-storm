#ifndef MED_IMAGING_ARITHMETRIC_VECTOR3F_H_
#define MED_IMAGING_ARITHMETRIC_VECTOR3F_H_

#include "MedImgArithmetic/mi_arithmetic_common.h"

MED_IMAGING_BEGIN_NAMESPACE

class Point3f;

class Arithmetic_Export Vector3f
{
public:
    union 
    {
        __m128 m_Vec128;
        struct 
        {
            float x, y, z;
        }m_Value;
        float _m[4];
    };

public:
     Vector3f()
    {
        m_Vec128 = _mm_setzero_ps();
    }

     Vector3f(const Vector3f& vec)
    {
        m_Vec128 = vec.m_Vec128;
    }

     Vector3f( const float _x, const float _y, const float _z )
    {
        m_Vec128 = _mm_setr_ps(_x, _y, _z, 0.0f);
    }

     Vector3f( const float scalar )
    {
        m_Vec128 = _mm_set1_ps(scalar);
    }

    Vector3f( const __m128 vf4 )
    {
        m_Vec128 = vf4;
    }

    inline Vector3f & operator =( const Vector3f &vec )
    {
        m_Vec128 = vec.m_Vec128;
        return *this;
    }

    inline Vector3f & SetX( const float x )
    {
        _vmathVfSetElement(m_Vec128, x, 0);
        return *this;
    }

    inline Vector3f & SetY( const float x )
    {
        _vmathVfSetElement(m_Vec128, x, 1);
        return *this;
    }

    inline Vector3f & SetZ( const float x )
    {
        _vmathVfSetElement(m_Vec128, x, 2);
        return *this;
    }

    inline const float GetX() const 
    {
        return _vmathVfGetElement(m_Vec128, 0);
        //return m_Value.x;
    }

    inline const float GetY() const 
    {
        return _vmathVfGetElement(m_Vec128, 1);
        // return m_Value.y;
    }

    inline const float GetZ() const 
    {
        return _vmathVfGetElement(m_Vec128, 2);
        // return m_Value.z;
    }

    inline Vector3f & SetElem( const int idx, const float value )
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

    inline const __m128 Get128() const
    {
        return m_Vec128;
    }

    inline Vector3f & Set128( __m128 vec)
    {
        m_Vec128 = vec;
        return *this;
    }

    inline const Vector3f operator+( const Vector3f &vec ) const
    {
        return Vector3f( _mm_add_ps( m_Vec128, vec.m_Vec128 ) );
    }

    inline Vector3f & operator +=( const Vector3f &vec )
    {
        m_Vec128 = _mm_add_ps( m_Vec128, vec.m_Vec128 );
        return *this;
    }

    inline const Vector3f operator -( const Vector3f &vec ) const
    {
        return Vector3f( _mm_sub_ps( m_Vec128, vec.m_Vec128 ) );
    }

    inline 
        Vector3f & operator -=( const Vector3f &vec )
    {
        m_Vec128 = _mm_sub_ps( m_Vec128, vec.m_Vec128 );
        return *this;
    }

    inline const Vector3f operator -( ) const
    {
        return Vector3f(_mm_sub_ps( _mm_setzero_ps(), m_Vec128 ) );
    }

    inline const Vector3f operator *( const float scalar ) const
    {
        return Vector3f( _mm_mul_ps( m_Vec128, _mm_set1_ps(scalar)) );
    }

    inline const Vector3f operator *( const Vector3f& vec ) const
    {
        return Vector3f( _mm_mul_ps( m_Vec128, vec.m_Vec128) );
    }

    inline Vector3f & operator *=( const float scalar )
    {
        m_Vec128 = _mm_mul_ps( m_Vec128, _mm_set1_ps(scalar));
        return *this;
    }

    inline const Vector3f operator /( const float scalar ) const
    {
        return Vector3f( _mm_div_ps( m_Vec128, _mm_set1_ps(scalar) ) );
    }

    inline const Vector3f operator /( const Vector3f& vec ) const
    {
        return Vector3f( _mm_div_ps( m_Vec128, vec.m_Vec128 ) );
    }

    inline Vector3f & operator /=( const float scalar )
    {
        m_Vec128 = _mm_div_ps( m_Vec128, _mm_set1_ps(scalar) );
        return *this;
    }

    inline Vector3f & operator /=( const Vector3f& vec )
    {
        m_Vec128 = _mm_div_ps( m_Vec128, vec.m_Vec128 );
        return *this;
    }

    inline bool operator != ( const Vector3f& vec) const
    {
        __m128 t = fabsf4(_mm_sub_ps(m_Vec128, vec.m_Vec128));
        t = _mm_add_ps(_mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)), vec_splat(t, 2));
        return _vmathVfGetElement(t , 0) > FLOAT_EPSILON;
    }

    inline bool operator == ( const Vector3f& vec) const
    {
        __m128 t = fabsf4(_mm_sub_ps(m_Vec128, vec.m_Vec128));
        t = _mm_add_ps(_mm_add_ps(vec_splat(t, 0), vec_splat(t, 1)), vec_splat(t, 2));
        return _vmathVfGetElement(t , 0) <= FLOAT_EPSILON;
    }

    inline float AngleBetween( const Vector3f& vec) const
    {
        float dlen = this->Magnitude() * vec.Magnitude();
        dlen = (dlen > FLOAT_EPSILON) ? dlen : FLOAT_EPSILON;
        float dprod = this->DotProduct(vec) / dlen;
        return acos(dprod);
    }

    inline Vector3f CrossProduct( const Vector3f& vec) const
    {
        return Vector3f( _vmathVfCross( m_Vec128, vec.m_Vec128 ) );
    }

    inline float DotProduct( const Vector3f& vec) const
    {
        __m128 t = _vmathVfDot3( m_Vec128, vec.m_Vec128 );
        return _vmathVfGetElement(t, 0);
    }

    inline float Magnitude() const
    {
        __m128 t = _mm_sqrt_ps(_vmathVfDot3( m_Vec128, m_Vec128 ));
        return _vmathVfGetElement(t, 0);
    }

    inline void Normalize()
    {
        m_Vec128 = _mm_mul_ps( m_Vec128, newtonrapson_rsqrt4( _vmathVfDot3( m_Vec128, m_Vec128 ) ) );
    }

    inline Vector3f GetNormalize() const
    {
        return Vector3f(_mm_mul_ps( m_Vec128, newtonrapson_rsqrt4( _vmathVfDot3( m_Vec128, m_Vec128 ) ) ));
    }

    inline Vector3f Reflect(const Vector3f& normal) const
    {
        return Vector3f(*this - (normal * 2.0f * this->DotProduct(normal)));
    }

    inline const float MaxElem() const
    {
        __m128 t = _mm_max_ps( _mm_max_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) ), vec_splat( m_Vec128, 2 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline const float MinElem() const
    {
        __m128 t = _mm_min_ps( _mm_min_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) ), vec_splat( m_Vec128, 2 ) );
        return _vmathVfGetElement(t , 0);
    }

     inline Vector3f MaxPerElem(const Vector3f& v) const
     {
         return Vector3f(_mm_max_ps(m_Vec128 , v.m_Vec128));
     }

     inline Vector3f MinPerElem(const Vector3f& v) const
     {
         return Vector3f(_mm_min_ps(m_Vec128 , v.m_Vec128));
     }

    inline const float Sum() const
    {
        __m128 t =  _mm_add_ps( _mm_add_ps( vec_splat( m_Vec128, 0 ), vec_splat( m_Vec128, 1 ) ), vec_splat( m_Vec128, 2 ) );
        return _vmathVfGetElement(t , 0);
    }

    inline const float Length() const
    {
        __m128 t = _mm_sqrt_ps(_vmathVfDot3( m_Vec128, m_Vec128 ));
        return _vmathVfGetElement(t, 0);
    }

    inline const float LengthSqr() const
    {
        __m128 t = _vmathVfDot3( m_Vec128, m_Vec128 );
        return _vmathVfGetElement(t, 0);
    }

    inline Vector3f LessThan(const Vector3f& v) const
    {
        return Vector3f(_mm_cmplt_ps(m_Vec128 , v.m_Vec128));
    }

    inline Vector3f ToAbs() const
    {
        return Vector3f(_mm_and_ps( m_Vec128, toM128( 0x7fffffff )));
    }

    inline void Abs()
    {
        m_Vec128 = _mm_and_ps( m_Vec128, toM128( 0x7fffffff ));
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Multiply two 3-D vectors per element
Arithmetic_Export  
    const Vector3f MulPerElem( const Vector3f &vec0, const Vector3f &vec1 );

// Divide two 3-D vectors per element
// Floating-point behavior matches standard library function divf4.
Arithmetic_Export  
    const Vector3f DivPerElem( const Vector3f &vec0, const Vector3f &vec1 );

// Compute the reciprocal of a 3-D vector per element
// Floating-point behavior matches standard library function recipf4.
Arithmetic_Export  
    const Vector3f RecipPerElem( const Vector3f &vec );

// Compute the absolute value of a 3-D vector per element
Arithmetic_Export  
    const Vector3f AbsPerElem( const Vector3f &vec );

// Maximum of two 3-D vectors per element
Arithmetic_Export  
    const Vector3f MaxPerElem( const Vector3f &vec0, const Vector3f &vec1 );

// Minimum of two 3-D vectors per element
Arithmetic_Export  
    const Vector3f MinPerElem( const Vector3f &vec0, const Vector3f &vec1 );

// Maximum element of a 3-D vector
Arithmetic_Export  
    const float MaxElem( const Vector3f &vec );

// Minimum element of a 3-D vector
Arithmetic_Export  
    const float MinElem( const Vector3f &vec );

// Compute the sum of all elements of a 3-D vector
Arithmetic_Export  
    const float Sum( const Vector3f &vec );

// Compute the dot product of two 3-D vectors
Arithmetic_Export  
    const float DotProduct( const Vector3f &vec0, const Vector3f &vec1 );

// Compute the square of the length of a 3-D vector
Arithmetic_Export  
    const float LengthSqr( const Vector3f &vec );

// Compute the length of a 3-D vector
Arithmetic_Export  
    const float Length( const Vector3f &vec );

// Normalize a 3-D vector, result is not accurate enough
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export  
    const Vector3f NormalizeApprox( const Vector3f &vec );

// Normalize a 3-D vector, using Newton iteration to refine rsqrt operation
// The result is unpredictable when all elements of vec are at or near zero.
Arithmetic_Export  
    const Vector3f Normalize( const Vector3f &vec );

// Compute cross product of two 3-D vectors
Arithmetic_Export  
    const Vector3f CrossProduct( const Vector3f &vec0, const Vector3f &vec1 );

// Linear interpolation between two 3-D vectors
// vec0 * (1 - t) + vec1 * t
Arithmetic_Export  
    const Vector3f Lerp( const float t, const Vector3f &vec0, const Vector3f &vec1 );

// Spherical linear interpolation between two 3-D vectors
// The result is unpredictable if the vectors point in opposite directions.
// Angle = acosf(dot( unitVec0, unitVec1 ));
// recipSinAngle = ( 1.0f / sinf( angle ) );
// scale0 = ( sinf( ( ( 1.0f - t ) * angle ) ) * recipSinAngle );
// scale1 = ( sinf( ( t * angle ) ) * recipSinAngle );
// return ( ( unitVec0 * scale0 ) + ( unitVec1 * scale1 ) );
Arithmetic_Export  
    const Vector3f Slerp( const float t, const Vector3f &unitVec0, const Vector3f &unitVec1 );

// Store x, y, and z elements of 3-D vector in first three words of a float ptr, preserving fourth word
Arithmetic_Export  
    void StoreXYZ( const Vector3f &vec, float * fptr );

// Load x, y, and z elements of 3-D vector in first three words of a float ptr, preserving fourth word
Arithmetic_Export  
    void LoadXYZ( Vector3f &vec, const float * fptr );

Arithmetic_Export const Vector3f operator /( const float scalar , const Vector3f& vec );


#ifdef _DEBUG
// Print a 3-D vector
// Function is only defined when _DEBUG is defined.
Arithmetic_Export  
    void print( const Vector3f &vec );

// Print a 3-D vector and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export  
    void print( const Vector3f &vec, const char * name );
#endif

MED_IMAGING_END_NAMESPACE
#endif
