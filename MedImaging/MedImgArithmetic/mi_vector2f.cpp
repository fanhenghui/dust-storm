#include "mi_vector2f.h"

MED_IMAGING_BEGIN_NAMESPACE

    const Vector2f MulPerElem( const Vector2f &vec0, const Vector2f &vec1 )
{
    return Vector2f(_mm_mul_ps(vec0.m_Vec128, vec1.m_Vec128));
}

 
    const Vector2f DivPerElem( const Vector2f &vec0, const Vector2f &vec1 )
{
    return Vector2f(_mm_div_ps(vec0.m_Vec128, vec1.m_Vec128));
}

 
    const Vector2f RecipPerElem( const Vector2f &vec )
{
    return Vector2f(_mm_rcp_ps(vec.m_Vec128));
}

 
    const Vector2f AbsPerElem( const Vector2f &vec )
{
    return Vector2f(fabsf4(vec.m_Vec128));
}

 
    const Vector2f MaxPerElem( const Vector2f &vec0, const Vector2f &vec1 )
{
    return Vector2f(_mm_max_ps(vec0.m_Vec128, vec1.m_Vec128));
}

 
    const Vector2f MinPerElem( const Vector2f &vec0, const Vector2f &vec1 )
{
    return Vector2f(_mm_min_ps(vec0.m_Vec128, vec1.m_Vec128));
}

 
    const float MaxElem( const Vector2f &vec )
{
    __m128 t = _mm_max_ps( vec_splat( vec.m_Vec128, 0 ), vec_splat( vec.m_Vec128, 1 ) );
    return _vmathVfGetElement(t , 0);
}

 
    const float MinElem( const Vector2f &vec )
{
    __m128 t = _mm_min_ps( vec_splat( vec.m_Vec128, 0 ), vec_splat( vec.m_Vec128, 1 ) );
    return _vmathVfGetElement(t , 0);
}

 
    const float Sum( const Vector2f &vec )
{
    __m128 t = _mm_add_ps( vec_splat( vec.m_Vec128, 0 ), vec_splat( vec.m_Vec128, 1 ) );
    return _vmathVfGetElement(t , 0);
}

 
    const float DotProduct( const Vector2f &vec0, const Vector2f &vec1 )
{
    __m128 t = _vmathVfDot2( vec0.m_Vec128, vec1.m_Vec128 );
    return _vmathVfGetElement(t, 0);
}

 
    const float LengthSqr( const Vector2f &vec )
{
    __m128 t = _vmathVfDot2( vec.m_Vec128, vec.m_Vec128 );
    return _vmathVfGetElement(t, 0);
}

 
    const float Length( const Vector2f &vec )
{
    __m128 t = _mm_sqrt_ps(_vmathVfDot2( vec.m_Vec128, vec.m_Vec128 ));
    return _vmathVfGetElement(t, 0);
}

 
    const Vector2f NormalizeApprox( const Vector2f &vec )
{
    return Vector2f( _mm_mul_ps( vec.m_Vec128, _mm_rsqrt_ps( _vmathVfDot2( vec.m_Vec128, vec.m_Vec128 ) ) ) );
}

 
    const Vector2f Normalize( const Vector2f &vec )
{
    return Vector2f( _mm_mul_ps( vec.m_Vec128, newtonrapson_rsqrt4( _vmathVfDot2( vec.m_Vec128, vec.m_Vec128 ) ) ) );
}

 
    const Vector2f Lerp( const float t, const Vector2f &vec0, const Vector2f &vec1 )
{
    return ( vec0 + ( ( vec1 - vec0 ) * t ) );
}

 
    const Vector2f Slerp( const float t, const Vector2f &unitVec0, const Vector2f &unitVec1 )
{
#define _MCSF_3D_SLERP_TOL 0.999f
    __m128 scales, scale0, scale1, cosAngle, angle, tttt, oneMinusT, angles, sines;
    cosAngle = _vmathVfDot2( unitVec0.m_Vec128, unitVec1.m_Vec128 );
    __m128 selectMask = _mm_cmpgt_ps( _mm_set1_ps(_MCSF_3D_SLERP_TOL), cosAngle );
    angle = acosf4( cosAngle );
    tttt = _mm_set1_ps(t);
    oneMinusT = _mm_sub_ps( _mm_set1_ps(1.0f), tttt );
    angles = _mm_unpacklo_ps( _mm_set1_ps(1.0f), tttt ); // angles = 1, t, 1, t
    angles = _mm_unpacklo_ps( angles, oneMinusT );		// angles = 1, 1-t, t, 1-t
    angles = _mm_mul_ps( angles, angle );
    sines = sinf4( angles );
    scales = _mm_div_ps( sines, vec_splat( sines, 0 ) );
    scale0 = vec_sel( oneMinusT, vec_splat( scales, 1 ), selectMask );
    scale1 = vec_sel( tttt, vec_splat( scales, 2 ), selectMask );
    return Vector2f( vec_madd( unitVec0.m_Vec128, scale0, _mm_mul_ps( unitVec1.m_Vec128, scale1 ) ) );
}

 
    void StoreXY( const Vector2f &vec, float * fptr )
{
    fptr[0] = vec.GetX();
    fptr[1] = vec.GetY();
}

 
    void LoadXY( Vector2f &vec, const float * fptr )
{
    vec = Vector2f(fptr[0], fptr[1]);
}

#ifdef _DEBUG

 
    void print( const Vector2f &vec )
{
    union { __m128 v; float s[2]; } tmp;
    tmp.v = vec.m_Vec128;
    printf( "( %f %f )\n", tmp.s[0], tmp.s[1] );
}

 
    void print( const Vector2f &vec, const char * name )
{
    union { __m128 v; float s[2]; } tmp;
    tmp.v = vec.m_Vec128;
    printf( "%s: ( %f %f )\n", name, tmp.s[0], tmp.s[1] );
}
#endif 

MED_IMAGING_END_NAMESPACE