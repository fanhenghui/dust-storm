
#include "mi_vector4f.h"

MED_IMAGING_BEGIN_NAMESPACE
 
    const Vector4f mul_per_elem( const Vector4f &vec0, const Vector4f &vec1 )
{
    return Vector4f( _mm_mul_ps( vec0._m128, vec1._m128 ) );
}

 
    const Vector4f div_per_elem( const Vector4f &vec0, const Vector4f &vec1 )
{
    return Vector4f( _mm_div_ps( vec0._m128, vec1._m128 ) );
}

 
    const Vector4f recip_per_elem( const Vector4f &vec )
{
    return Vector4f( _mm_rcp_ps( vec._m128 ) );
}

 
    const Vector4f abs_per_elem( const Vector4f &vec )
{
    return Vector4f( fabsf4( vec._m128 ) );
}

 
    const Vector4f max_per_elem( const Vector4f &vec0, const Vector4f &vec1 )
{
    return Vector4f( _mm_max_ps( vec0._m128, vec1._m128 ) );
}

 
    const Vector4f min_per_elem( const Vector4f &vec0, const Vector4f &vec1 )
{
    return Vector4f( _mm_min_ps( vec0._m128, vec1._m128 ) );
}

 
    const float max_elem( const Vector4f &vec )
{
    __m128 t = _mm_max_ps(
        _mm_max_ps( vec_splat( vec._m128, 0 ), vec_splat( vec._m128, 1 ) ),
        _mm_max_ps( vec_splat( vec._m128, 2 ), vec_splat( vec._m128, 3 ) ) );
    return _vmathVfGetElement(t , 0); 
}

 
    const float min_elem( const Vector4f &vec )
{
    __m128 t = _mm_min_ps(
        _mm_min_ps( vec_splat( vec._m128, 0 ), vec_splat( vec._m128, 1 ) ),
        _mm_min_ps( vec_splat( vec._m128, 2 ), vec_splat( vec._m128, 3 ) ) );
    return _vmathVfGetElement(t , 0); 
}

 
    const float sum( const Vector4f &vec )
{
    __m128 t = _mm_add_ps(
        _mm_add_ps( vec_splat( vec._m128, 0 ), vec_splat( vec._m128, 1 ) ),
        _mm_add_ps( vec_splat( vec._m128, 2 ), vec_splat( vec._m128, 3 ) ) );
    return _vmathVfGetElement(t , 0); 
}

 
    const float dot( const Vector4f &vec0, const Vector4f &vec1 )
{
    __m128 t = _vmathVfDot4( vec0._m128, vec1._m128 );
    return _vmathVfGetElement(t , 0); 
}

 
    const float length_sqr( const Vector4f &vec )
{
    __m128 temp = _vmathVfDot4( vec._m128, vec._m128 );
    return _vmathVfGetElement(temp, 0);
}

 
    const float length( const Vector4f &vec )
{
    __m128 temp = _mm_sqrt_ps(_vmathVfDot4( vec._m128, vec._m128 ));
    return _vmathVfGetElement(temp, 0);
}

 
    const Vector4f normalize_approx( const Vector4f &vec )
{
    return Vector4f( _mm_mul_ps( vec._m128, _mm_rsqrt_ps( _vmathVfDot4( vec._m128, vec._m128 ) ) ) );
}

 
    const Vector4f normalize( const Vector4f &vec )
{
    return Vector4f( _mm_mul_ps( vec._m128, newtonrapson_rsqrt4( _vmathVfDot4( vec._m128, vec._m128 ) ) ) );
}

 
    const Vector4f lerp( const float t, const Vector4f &vec0, const Vector4f &vec1 )
{
    return ( vec0 + ( ( vec1 - vec0 ) * t ) );
}

 
    const Vector4f slerp( const float t, const Vector4f &unitVec0, const Vector4f &unitVec1 )
{
#define _MCSF_3D_SLERP_TOL 0.999f
    __m128 scales, scale0, scale1, cosAngle, angle, tttt, oneMinusT, angles, sines;
    cosAngle = _vmathVfDot4( unitVec0._m128, unitVec1._m128 );
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
    return Vector4f( vec_madd( unitVec0._m128, scale0, _mm_mul_ps( unitVec1._m128, scale1 ) ) );
}

 
    void store_xyzw( const Vector4f &vec, float * fptr )
{
    fptr[0] = vec._value.x;
    fptr[1] = vec._value.y;
    fptr[2] = vec._value.z;
    fptr[3] = vec._value.w;
}

 
    void load_xyzw( Vector4f &vec, const float * fptr )
{
    vec = Vector4f(fptr[0], fptr[1], fptr[2], fptr[3]);
}

#ifdef _DEBUG

 
    void print( const Vector4f &vec )
{
    union { __m128 v; float s[4]; } tmp;
    tmp.v = vec._m128;
    printf( "( %f %f %f %f )\n", tmp.s[0], tmp.s[1], tmp.s[2], tmp.s[3] );
}

 
    void print( const Vector4f &vec, const char * name )
{
    union { __m128 v; float s[4]; } tmp;
    tmp.v = vec._m128;
    printf( "%s: ( %f %f %f %f )\n", name, tmp.s[0], tmp.s[1], tmp.s[2], tmp.s[3] );
}

#endif

MED_IMAGING_END_NAMESPACE