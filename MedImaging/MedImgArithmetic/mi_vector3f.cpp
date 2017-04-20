#include "mi_vector3f.h"

MED_IMAGING_BEGIN_NAMESPACE

     
    const Vector3f mul_per_elem( const Vector3f &vec0, const Vector3f &vec1 )
{
    return Vector3f( _mm_mul_ps( vec0.m_Vec128, vec1.m_Vec128 ) );
}

 
    const Vector3f div_per_elem( const Vector3f &vec0, const Vector3f &vec1 )
{
    return Vector3f( _mm_div_ps( vec0.m_Vec128, vec1.m_Vec128 ) );
}

 
    const Vector3f recip_per_elem( const Vector3f &vec )
{
    return Vector3f( _mm_rcp_ps( vec.m_Vec128 ) );
}

 
    const Vector3f abs_per_elem( const Vector3f &vec )
{
    return Vector3f( fabsf4( vec.m_Vec128 ) );
}

 
    const Vector3f max_per_elem( const Vector3f &vec0, const Vector3f &vec1 )
{
    return Vector3f( _mm_max_ps( vec0.m_Vec128, vec1.m_Vec128 ) );
}

 
    const Vector3f min_per_elem( const Vector3f &vec0, const Vector3f &vec1 )
{
    return Vector3f( _mm_min_ps( vec0.m_Vec128, vec1.m_Vec128 ) );
}

 
    const float max_elem( const Vector3f &vec )
{
    __m128 t = _mm_max_ps( _mm_max_ps( vec_splat( vec.m_Vec128, 0 ), vec_splat( vec.m_Vec128, 1 ) ), vec_splat( vec.m_Vec128, 2 ) );
    return _vmathVfGetElement(t , 0);
}

 
    const float min_elem( const Vector3f &vec )
{
    __m128 t = _mm_min_ps( _mm_min_ps( vec_splat( vec.m_Vec128, 0 ), vec_splat( vec.m_Vec128, 1 ) ), vec_splat( vec.m_Vec128, 2 ) );
    return _vmathVfGetElement(t , 0);
}

 
    const float sum( const Vector3f &vec )
{
    __m128 t =  _mm_add_ps( _mm_add_ps( vec_splat( vec.m_Vec128, 0 ), vec_splat( vec.m_Vec128, 1 ) ), vec_splat( vec.m_Vec128, 2 ) );
    return _vmathVfGetElement(t , 0);
}

 
    const float dot_product( const Vector3f &vec0, const Vector3f &vec1 )
{
    __m128 t = _vmathVfDot3( vec0.m_Vec128, vec1.m_Vec128 );
    return _vmathVfGetElement(t, 0);
}

 
    const float length_sqr( const Vector3f &vec )
{
    __m128 t = _vmathVfDot3( vec.m_Vec128, vec.m_Vec128 );
    return _vmathVfGetElement(t, 0);
}

 
    const float length( const Vector3f &vec )
{
    __m128 t = _mm_sqrt_ps(_vmathVfDot3( vec.m_Vec128, vec.m_Vec128 ));
    return _vmathVfGetElement(t, 0);
}

 
    const Vector3f normalize_approx( const Vector3f &vec )
{
    return Vector3f( _mm_mul_ps( vec.m_Vec128, _mm_rsqrt_ps( _vmathVfDot3( vec.m_Vec128, vec.m_Vec128 ) ) ) );
}

 
    const Vector3f normalize( const Vector3f &vec )
{
    return Vector3f( _mm_mul_ps( vec.m_Vec128, newtonrapson_rsqrt4( _vmathVfDot3( vec.m_Vec128, vec.m_Vec128 ) ) ) );
}

 
    const Vector3f cross( const Vector3f &vec0, const Vector3f &vec1 )
{
    return Vector3f( _vmathVfCross( vec0.m_Vec128, vec1.m_Vec128 ) );
}

 
    const Vector3f lerp( const float t, const Vector3f &vec0, const Vector3f &vec1 )
{
    return ( vec0 + ( ( vec1 - vec0 ) * t ) );
}

 
    const Vector3f slerp( const float t, const Vector3f &unitVec0, const Vector3f &unitVec1 )
{
#define _MCSF_3D_SLERP_TOL 0.999f
    __m128 scales, scale0, scale1, cosAngle, angle, tttt, oneMinusT, angles, sines;
    cosAngle = _vmathVfDot3( unitVec0.m_Vec128, unitVec1.m_Vec128 );
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
    return Vector3f( vec_madd( unitVec0.m_Vec128, scale0, _mm_mul_ps( unitVec1.m_Vec128, scale1 ) ) );
}


 
    void store_xyz(const Vector3f& vec, float* fptr)
{
    fptr[0] = vec.m_Value.x;
    fptr[1] = vec.m_Value.y;
    fptr[2] = vec.m_Value.z;
}

 
    void load_xyz(Vector3f& vec, float* fptr)
{
    vec = Vector3f(fptr[0], fptr[1], fptr[2]);
}

#ifdef _DEBUG
 
    void print( const Vector3f &vec )
{
    union { __m128 v; float s[4]; } tmp;
    tmp.v = vec.m_Vec128;
    printf( "( %f %f %f )\n", tmp.s[0], tmp.s[1], tmp.s[2] );
}

 
    void print( const Vector3f &vec, const char * name )
{
    union { __m128 v; float s[4]; } tmp;
    tmp.v = vec.m_Vec128;
    printf( "%s: ( %f %f %f )\n", name, tmp.s[0], tmp.s[1], tmp.s[2] );
}

    const Vector3f operator/(const float scalar , const Vector3f& vec)
    {
        return Vector3f( _mm_div_ps(  _mm_set1_ps(scalar) , vec.m_Vec128) );
    }

#endif

MED_IMAGING_END_NAMESPACE
