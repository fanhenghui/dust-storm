#ifndef MED_IMAGING_ARITHMETIC_COMMON_H_
#define MED_IMAGING_ARITHMETIC_COMMON_H_
 
#include "MedImgArithmetic/mi_arithmetic_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

#include <math.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <assert.h>
#include <stdio.h>

#if (defined (_WIN32) && (_MSC_VER) && _MSC_VER >= 1400)
#define USE_SSE3_LDDQU

#define MCSF_3D_ALIGNED_CLASS16(a) __declspec(align(16)) a
#define MCSF_3D_ALIGN16 __declspec(align(16))
#define MCSF3DARITHMETIC_FORCE_INLINE inline //__forceinline 
#else
#define MCSF_3D_ALIGNED_CLASS16(a) a __attribute__ ((aligned (16)))	
#define MCSF_3D_ALIGN16 __attribute__ ((aligned (16)))	
#define MCSF3DARITHMETIC_FORCE_INLINE inline __attribute__ ((always_inline))
#ifdef __SSE3__
#define USE_SSE3_LDDQU
#endif //__SSE3__
#endif//_WIN32

#ifdef USE_SSE3_LDDQU
#include <pmmintrin.h>//_mm_lddqu_si128
#endif //USE_SSE3_LDDQU


typedef __m128 vec_float4;
typedef __m128 vec_uint4;
typedef __m128 vec_int4;
typedef __m128i vec_uchar16;
typedef __m128i vec_ushort8;

//-----------------------------------------------------------------------------
// Constants
// for permutes words are labeled [x,y,z,w] [a,b,c,d]
#define _VECTORMATH_PERM_X 0x00010203
#define _VECTORMATH_PERM_Y 0x04050607
#define _VECTORMATH_PERM_Z 0x08090a0b
#define _VECTORMATH_PERM_W 0x0c0d0e0f
#define _VECTORMATH_PERM_A 0x10111213
#define _VECTORMATH_PERM_B 0x14151617
#define _VECTORMATH_PERM_C 0x18191a1b
#define _VECTORMATH_PERM_D 0x1c1d1e1f
#define _VECTORMATH_PERM_XYZA (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_X, _VECTORMATH_PERM_Y, _VECTORMATH_PERM_Z, _VECTORMATH_PERM_A }
#define _VECTORMATH_PERM_ZXYW (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_Z, _VECTORMATH_PERM_X, _VECTORMATH_PERM_Y, _VECTORMATH_PERM_W }
#define _VECTORMATH_PERM_YZXW (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_Y, _VECTORMATH_PERM_Z, _VECTORMATH_PERM_X, _VECTORMATH_PERM_W }
#define _VECTORMATH_PERM_YZAB (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_Y, _VECTORMATH_PERM_Z, _VECTORMATH_PERM_A, _VECTORMATH_PERM_B }
#define _VECTORMATH_PERM_ZABC (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_Z, _VECTORMATH_PERM_A, _VECTORMATH_PERM_B, _VECTORMATH_PERM_C }
#define _VECTORMATH_PERM_XYAW (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_X, _VECTORMATH_PERM_Y, _VECTORMATH_PERM_A, _VECTORMATH_PERM_W }
#define _VECTORMATH_PERM_XAZW (vec_uchar16)(vec_uint4){ _VECTORMATH_PERM_X, _VECTORMATH_PERM_A, _VECTORMATH_PERM_Z, _VECTORMATH_PERM_W }
#define _VECTORMATH_MASK_0xF000 (vec_uint4){ 0xffffffff, 0, 0, 0 }
#define _VECTORMATH_MASK_0x0F00 (vec_uint4){ 0, 0xffffffff, 0, 0 }
#define _VECTORMATH_MASK_0x00F0 (vec_uint4){ 0, 0, 0xffffffff, 0 }
#define _VECTORMATH_MASK_0x000F (vec_uint4){ 0, 0, 0, 0xffffffff }
#define _VECTORMATH_UNIT_1000 _mm_setr_ps(1.0f,0.0f,0.0f,0.0f) // (__m128){ 1.0f, 0.0f, 0.0f, 0.0f }
#define _VECTORMATH_UNIT_0100 _mm_setr_ps(0.0f,1.0f,0.0f,0.0f) // (__m128){ 0.0f, 1.0f, 0.0f, 0.0f }
#define _VECTORMATH_UNIT_0010 _mm_setr_ps(0.0f,0.0f,1.0f,0.0f) // (__m128){ 0.0f, 0.0f, 1.0f, 0.0f }
#define _VECTORMATH_UNIT_0001 _mm_setr_ps(0.0f,0.0f,0.0f,1.0f) // (__m128){ 0.0f, 0.0f, 0.0f, 1.0f }
#define _VECTORMATH_SLERP_TOL 0.999f

//-----------------------------------------------------------------------------

#define vec_splat(x, e) _mm_shuffle_ps(x, x, _MM_SHUFFLE(e,e,e,e))
 
#define _mm_ror_ps(vec,i)	\
	(((i)%4) ? (_mm_shuffle_ps(vec,vec, _MM_SHUFFLE((unsigned char)(i+3)%4,(unsigned char)(i+2)%4,(unsigned char)(i+1)%4,(unsigned char)(i+0)%4))) : (vec))
#define _mm_rol_ps(vec,i)	\
	(((i)%4) ? (_mm_shuffle_ps(vec,vec, _MM_SHUFFLE((unsigned char)(7-i)%4,(unsigned char)(6-i)%4,(unsigned char)(5-i)%4,(unsigned char)(4-i)%4))) : (vec))

#define vec_sld(vec,vec2,x) _mm_ror_ps(vec, ((x)/4))

#define _mm_abs_ps(vec)		_mm_andnot_ps(_MASKSIGN_,vec)
#define _mm_neg_ps(vec)		_mm_xor_ps(_MASKSIGN_,vec)

#define vec_madd(a, b, c) _mm_add_ps(c, _mm_mul_ps(a, b) )

#define vec_nmsub(a,b,c) _mm_sub_ps( c, _mm_mul_ps( a, b ) )
#define vec_sub(a,b) _mm_sub_ps( a, b )
#define vec_add(a,b) _mm_add_ps( a, b )
#define vec_mul(a,b) _mm_mul_ps( a, b )
#define vec_xor(a,b) _mm_xor_ps( a, b )
#define vec_and(a,b) _mm_and_ps( a, b )
#define vec_cmpeq(a,b) _mm_cmpeq_ps( a, b )
#define vec_cmpgt(a,b) _mm_cmpgt_ps( a, b )

#define vec_mergeh(a,b) _mm_unpacklo_ps( a, b )
#define vec_mergel(a,b) _mm_unpackhi_ps( a, b )

#define vec_andc(a,b) _mm_andnot_ps( b, a )

#define sqrtf4(x) _mm_sqrt_ps( x )
#define rsqrtf4(x) _mm_rsqrt_ps( x )
#define recipf4(x) _mm_rcp_ps( x )
#define negatef4(x) _mm_sub_ps( _mm_setzero_ps(), x )

#define _vmathVfSetElement(vec, scalar, slot) ((float *)&(vec))[slot] = scalar
#define _vmathVfGetElement(vec , slot) ((float*)(&vec))[slot]

static  __m128 vec_sel(__m128 a, __m128 b, __m128 mask)
{
	return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}
static  __m128 vec_sel(__m128 a, __m128 b, const unsigned int *_mask)
{
	return vec_sel(a, b, _mm_load_ps((float *)_mask));
}
static  __m128 vec_sel(__m128 a, __m128 b, unsigned int _mask)
{
	return vec_sel(a, b, _mm_set1_ps(*(float *)&_mask));
}

static  __m128 toM128(unsigned int x)
{
	return _mm_set1_ps( *(float *)&x );
}

static  __m128 fabsf4(__m128 x)
{
	return _mm_and_ps( x, toM128( 0x7fffffff ) );
}

static  __m128 vec_cts(__m128 x)
{
	__m128i result = _mm_cvtps_epi32(x);
    return (__m128 &)result;
}

static  __m128 vec_ctf(__m128 x)
{
	return _mm_cvtepi32_ps((__m128i &)x);
}

static  __m128 vec_round_to_int(__m128 x)
{
	__m128i result = _mm_cvtps_epi32(x);
	return (__m128 &)result;
}

static  __m128 vec_truncate_to_int(__m128 x)
{
	__m128i result = _mm_cvttps_epi32(x);
	return (__m128 &)result;
}

static  __m128 newtonrapson_rsqrt4( const __m128 v )
{   
#define _half4 _mm_setr_ps(.5f,.5f,.5f,.5f) 
#define _three _mm_setr_ps(3.f,3.f,3.f,3.f)
	const __m128 approx = _mm_rsqrt_ps( v );   
	const __m128 muls = _mm_mul_ps(_mm_mul_ps(v, approx), approx);   
	return _mm_mul_ps(_mm_mul_ps(_half4, approx), _mm_sub_ps(_three, muls) );
}

static  __m128 acosf4(__m128 x)
{
    __m128 xabs = fabsf4(x);
	__m128 select = _mm_cmplt_ps( x, _mm_setzero_ps() );
    __m128 t1 = sqrtf4(vec_sub(_mm_set1_ps(1.0f), xabs));
    
    /* Instruction counts can be reduced if the polynomial was
     * computed entirely from nested (dependent) fma's. However, 
     * to reduce the number of pipeline stalls, the polygon is evaluated 
     * in two halves (hi amd lo). 
     */
    __m128 xabs2 = _mm_mul_ps(xabs,  xabs);
    __m128 xabs4 = _mm_mul_ps(xabs2, xabs2);
    __m128 hi = vec_madd(vec_madd(vec_madd(_mm_set1_ps(-0.0012624911f),
		xabs, _mm_set1_ps(0.0066700901f)),
			xabs, _mm_set1_ps(-0.0170881256f)),
				xabs, _mm_set1_ps( 0.0308918810f));
    __m128 lo = vec_madd(vec_madd(vec_madd(_mm_set1_ps(-0.0501743046f),
		xabs, _mm_set1_ps(0.0889789874f)),
			xabs, _mm_set1_ps(-0.2145988016f)),
				xabs, _mm_set1_ps( 1.5707963050f));
    
    __m128 result = vec_madd(hi, xabs4, lo);
    
    // Adjust the result if x is negactive.
    return vec_sel(
		vec_mul(t1, result),									// Positive
		vec_nmsub(t1, result, _mm_set1_ps(3.1415926535898f)),	// Negative
		select);
}

static  __m128 sinf4(vec_float4 x)
{

//
// Common constants used to evaluate sinf4/cosf4/tanf4
//
#define _SINCOS_CC0  -0.0013602249f
#define _SINCOS_CC1   0.0416566950f
#define _SINCOS_CC2  -0.4999990225f
#define _SINCOS_SC0  -0.0001950727f
#define _SINCOS_SC1   0.0083320758f
#define _SINCOS_SC2  -0.1666665247f

#define _SINCOS_KC1  1.57079625129f
#define _SINCOS_KC2  7.54978995489e-8f

    vec_float4 xl,xl2,xl3,res;

    // Range reduction using : xl = angle * TwoOverPi;
    //  
    xl = vec_mul(x, _mm_set1_ps(0.63661977236f));

    // Find the quadrant the angle falls in
    // using:  q = (int) (ceil(abs(xl))*sign(xl))
    //
    vec_int4 q = vec_cts(xl);

    // Compute an offset based on the quadrant that the angle falls in
    // 
    vec_int4 offset = _mm_and_ps(q,toM128(0x3));

    // Remainder in range [-pi/4..pi/4]
    //
    vec_float4 qf = vec_ctf(q);
    xl  = vec_nmsub(qf,_mm_set1_ps(_SINCOS_KC2),vec_nmsub(qf,_mm_set1_ps(_SINCOS_KC1),x));
    
    // Compute x^2 and x^3
    //
    xl2 = vec_mul(xl,xl);
    xl3 = vec_mul(xl2,xl);
    
    // Compute both the sin and cos of the angles
    // using a polynomial expression:
    //   cx = 1.0f + xl2 * ((C0 * xl2 + C1) * xl2 + C2), and
    //   sx = xl + xl3 * ((S0 * xl2 + S1) * xl2 + S2)
    //
    
    vec_float4 cx =
		vec_madd(
			vec_madd(
				vec_madd(_mm_set1_ps(_SINCOS_CC0),xl2,_mm_set1_ps(_SINCOS_CC1)),xl2,_mm_set1_ps(_SINCOS_CC2)),xl2,_mm_set1_ps(1.0f));
    vec_float4 sx =
		vec_madd(
			vec_madd(
				vec_madd(_mm_set1_ps(_SINCOS_SC0),xl2,_mm_set1_ps(_SINCOS_SC1)),xl2,_mm_set1_ps(_SINCOS_SC2)),xl3,xl);

    // Use the cosine when the offset is odd and the sin
    // when the offset is even
    //
    res = vec_sel(cx,sx,vec_cmpeq(vec_and(offset,
                                          toM128(0x1)),
										  _mm_setzero_ps()));

    // Flip the sign of the result when (offset mod 4) = 1 or 2
    //
    return vec_sel(
		vec_xor(toM128(0x80000000U), res),	// Negative
		res,								// Positive
		vec_cmpeq(vec_and(offset,toM128(0x2)),_mm_setzero_ps()));
}

static  void sincosf4(vec_float4 x, vec_float4* s, vec_float4* c)
{
    vec_float4 xl,xl2,xl3;
    vec_int4   offsetSin, offsetCos;

    // Range reduction using : xl = angle * TwoOverPi;
    //  
    xl = vec_mul(x, _mm_set1_ps(0.63661977236f));

    // Find the quadrant the angle falls in
    // using:  q = (int) (ceil(abs(xl))*sign(xl))
    //
    //vec_int4 q = vec_cts(vec_add(xl,vec_sel(_mm_set1_ps(0.5f),xl,(0x80000000))));
    vec_int4 q = vec_cts(xl);
     
    // Compute the offset based on the quadrant that the angle falls in.
    // Add 1 to the offset for the cosine. 
    //
    offsetSin = vec_and(q,toM128((int)0x3));
	__m128i temp = _mm_add_epi32(_mm_set1_epi32(1),(__m128i &)offsetSin);
	offsetCos = (__m128 &)temp;

    // Remainder in range [-pi/4..pi/4]
    //
    vec_float4 qf = vec_ctf(q);
    xl  = vec_nmsub(qf,_mm_set1_ps(_SINCOS_KC2),vec_nmsub(qf,_mm_set1_ps(_SINCOS_KC1),x));
    
    // Compute x^2 and x^3
    //
    xl2 = vec_mul(xl,xl);
    xl3 = vec_mul(xl2,xl);
    
    // Compute both the sin and cos of the angles
    // using a polynomial expression:
    //   cx = 1.0f + xl2 * ((C0 * xl2 + C1) * xl2 + C2), and
    //   sx = xl + xl3 * ((S0 * xl2 + S1) * xl2 + S2)
    //
    vec_float4 cx =
		vec_madd(
			vec_madd(
				vec_madd(_mm_set1_ps(_SINCOS_CC0),xl2,_mm_set1_ps(_SINCOS_CC1)),xl2,_mm_set1_ps(_SINCOS_CC2)),xl2,_mm_set1_ps(1.0f));
    vec_float4 sx =
		vec_madd(
			vec_madd(
				vec_madd(_mm_set1_ps(_SINCOS_SC0),xl2,_mm_set1_ps(_SINCOS_SC1)),xl2,_mm_set1_ps(_SINCOS_SC2)),xl3,xl);

    // Use the cosine when the offset is odd and the sin
    // when the offset is even
    //
    vec_uint4 sinMask = (vec_uint4)vec_cmpeq(vec_and(offsetSin,toM128(0x1)),_mm_setzero_ps());
    vec_uint4 cosMask = (vec_uint4)vec_cmpeq(vec_and(offsetCos,toM128(0x1)),_mm_setzero_ps());    
    *s = vec_sel(cx,sx,sinMask);
    *c = vec_sel(cx,sx,cosMask);

    // Flip the sign of the result when (offset mod 4) = 1 or 2
    //
    sinMask = vec_cmpeq(vec_and(offsetSin,toM128(0x2)),_mm_setzero_ps());
    cosMask = vec_cmpeq(vec_and(offsetCos,toM128(0x2)),_mm_setzero_ps());
    
    *s = vec_sel((vec_float4)vec_xor(toM128(0x80000000),(vec_uint4)*s),*s,sinMask);
    *c = vec_sel((vec_float4)vec_xor(toM128(0x80000000),(vec_uint4)*c),*c,cosMask);    
}

#define     _vmath_shufps(a, b, immx, immy, immz, immw) _mm_shuffle_ps(a, b, _MM_SHUFFLE(immw, immz, immy, immx))

static  __m128 _vmathVfDot2( __m128 vec0, __m128 vec1 )
{
    __m128 result = _mm_mul_ps( vec0, vec1);
    return _mm_add_ps( vec_splat( result, 0 ), vec_splat( result, 1 ) );
}

static  __m128 _vmathVfDot3( __m128 vec0, __m128 vec1 )
{
	__m128 result = _mm_mul_ps( vec0, vec1);
	return _mm_add_ps( vec_splat( result, 0 ), _mm_add_ps( vec_splat( result, 1 ), vec_splat( result, 2 ) ) );
}

static  __m128 _vmathVfDot4( __m128 vec0, __m128 vec1 )
{
	__m128 result = _mm_mul_ps(vec0, vec1);
	return _mm_add_ps(_mm_shuffle_ps(result, result, _MM_SHUFFLE(0,0,0,0)),
		_mm_add_ps(_mm_shuffle_ps(result, result, _MM_SHUFFLE(1,1,1,1)),
		_mm_add_ps(_mm_shuffle_ps(result, result, _MM_SHUFFLE(2,2,2,2)), _mm_shuffle_ps(result, result, _MM_SHUFFLE(3,3,3,3)))));
}

static  __m128 _vmathVfCross( __m128 vec0, __m128 vec1 )
{
	__m128 tmp0, tmp1, tmp2, tmp3, result;
	tmp0 = _mm_shuffle_ps( vec0, vec0, _MM_SHUFFLE(3,0,2,1) );
	tmp1 = _mm_shuffle_ps( vec1, vec1, _MM_SHUFFLE(3,1,0,2) );
	tmp2 = _mm_shuffle_ps( vec0, vec0, _MM_SHUFFLE(3,1,0,2) );
	tmp3 = _mm_shuffle_ps( vec1, vec1, _MM_SHUFFLE(3,0,2,1) );
	result = vec_mul( tmp0, tmp1 );
	result = vec_nmsub( tmp2, tmp3, result );
	return result;
}

static  bool _isAlignedForSSE(const void *p)
{
    return (((size_t)p) & 15) == 0;
}

MED_IMAGING_END_NAMESPACE

#endif
