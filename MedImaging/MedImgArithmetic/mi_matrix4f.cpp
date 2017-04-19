#include "mi_matrix4f.h"

MED_IMAGING_BEGIN_NAMESPACE

     
    const Matrix4f AppendScale( const Matrix4f & mat, const Vector3f &scaleVec )
{
    return Matrix4f(
        ( mat.GetCol0() * scaleVec.GetX( ) ),
        ( mat.GetCol1() * scaleVec.GetY( ) ),
        ( mat.GetCol2() * scaleVec.GetZ( ) ),
        mat.GetCol3()
        );
}

 
    const Matrix4f PrependScale( const Vector3f &scaleVec, const Matrix4f & mat )
{
    Vector4f scale4 = Vector4f( scaleVec, 1.0f );
    return Matrix4f(
        MulPerElem( mat.GetCol0(), scale4 ),
        MulPerElem( mat.GetCol1(), scale4 ),
        MulPerElem( mat.GetCol2(), scale4 ),
        MulPerElem( mat.GetCol3(), scale4 )
        );
}

 
    const Matrix4f MulPerElem( const Matrix4f & mat0, const Matrix4f & mat1 )
{
    return Matrix4f(
        MulPerElem( mat0.GetCol0(), mat1.GetCol0() ),
        MulPerElem( mat0.GetCol1(), mat1.GetCol1() ),
        MulPerElem( mat0.GetCol2(), mat1.GetCol2() ),
        MulPerElem( mat0.GetCol3(), mat1.GetCol3() )
        );
}

 
    const Matrix4f AbsPerElem( const Matrix4f & mat )
{
    return Matrix4f(
        AbsPerElem( mat.GetCol0() ),
        AbsPerElem( mat.GetCol1() ),
        AbsPerElem( mat.GetCol2() ),
        AbsPerElem( mat.GetCol3() )
        );
}

 
    const Matrix4f Transpose( const Matrix4f & mat )
{
    __m128 tmp0, tmp1, tmp2, tmp3, res0, res1, res2, res3;
    tmp0 = vec_mergeh( mat.GetCol0().m_Vec128, mat.GetCol2().m_Vec128 );
    tmp1 = vec_mergeh( mat.GetCol1().m_Vec128, mat.GetCol3().m_Vec128 );
    tmp2 = vec_mergel( mat.GetCol0().m_Vec128, mat.GetCol2().m_Vec128 );
    tmp3 = vec_mergel( mat.GetCol1().m_Vec128, mat.GetCol3().m_Vec128 );
    res0 = vec_mergeh( tmp0, tmp1 );
    res1 = vec_mergel( tmp0, tmp1 );
    res2 = vec_mergeh( tmp2, tmp3 );
    res3 = vec_mergel( tmp2, tmp3 );
    return Matrix4f(
        Vector4f( res0 ),
        Vector4f( res1 ),
        Vector4f( res2 ),
        Vector4f( res3 )
        );
}

 
    const Matrix4f Inverse( const Matrix4f & mat )
{
    __m128 Va,Vb,Vc;
    __m128 r1,r2,r3,tt,tt2;
    __m128 sum,Det,RDet;
    __m128 trns0,trns1,trns2,trns3;

    __m128 _L1 = mat.GetCol0().m_Vec128;
    __m128 _L2 = mat.GetCol1().m_Vec128;
    __m128 _L3 = mat.GetCol2().m_Vec128;
    __m128 _L4 = mat.GetCol3().m_Vec128;
    // Calculating the minterms for the first line.

    // _mm_ror_ps is just a macro using _mm_shuffle_ps().
    tt = _L4; tt2 = _mm_ror_ps(_L3,1); 
    Vc = _mm_mul_ps(tt2,_mm_ror_ps(tt,0));					// V3'dot V4
    Va = _mm_mul_ps(tt2,_mm_ror_ps(tt,2));					// V3'dot V4"
    Vb = _mm_mul_ps(tt2,_mm_ror_ps(tt,3));					// V3' dot V4^

    r1 = _mm_sub_ps(_mm_ror_ps(Va,1),_mm_ror_ps(Vc,2));		// V3" dot V4^ - V3^ dot V4"
    r2 = _mm_sub_ps(_mm_ror_ps(Vb,2),_mm_ror_ps(Vb,0));		// V3^ dot V4' - V3' dot V4^
    r3 = _mm_sub_ps(_mm_ror_ps(Va,0),_mm_ror_ps(Vc,1));		// V3' dot V4" - V3" dot V4'

    tt = _L2;
    Va = _mm_ror_ps(tt,1);		sum = _mm_mul_ps(Va,r1);
    Vb = _mm_ror_ps(tt,2);		sum = _mm_add_ps(sum,_mm_mul_ps(Vb,r2));
    Vc = _mm_ror_ps(tt,3);		sum = _mm_add_ps(sum,_mm_mul_ps(Vc,r3));

    // Calculating the determinant.
    Det = _mm_mul_ps(sum,_L1);
    Det = _mm_add_ps(Det,_mm_movehl_ps(Det,Det));

    MCSF_3D_ALIGN16 const unsigned int _vmathPNPN[4] = {0x00000000, 0x80000000, 0x00000000, 0x80000000};
    MCSF_3D_ALIGN16 const unsigned int _vmathNPNP[4] = {0x80000000, 0x00000000, 0x80000000, 0x00000000};
    const __m128 Sign_PNPN = _mm_load_ps((float *)_vmathPNPN);
    const __m128 Sign_NPNP = _mm_load_ps((float *)_vmathNPNP);

    __m128 mtL1 = _mm_xor_ps(sum,Sign_PNPN);

    // Calculating the minterms of the second line (using previous results).
    tt = _mm_ror_ps(_L1,1);		sum = _mm_mul_ps(tt,r1);
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r2));
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r3));
    __m128 mtL2 = _mm_xor_ps(sum,Sign_NPNP);

    // Testing the determinant.
    Det = _mm_sub_ss(Det,_mm_shuffle_ps(Det,Det,1));

    // Calculating the minterms of the third line.
    tt = _mm_ror_ps(_L1,1);
    Va = _mm_mul_ps(tt,Vb);									// V1' dot V2"
    Vb = _mm_mul_ps(tt,Vc);									// V1' dot V2^
    Vc = _mm_mul_ps(tt,_L2);								// V1' dot V2

    r1 = _mm_sub_ps(_mm_ror_ps(Va,1),_mm_ror_ps(Vc,2));		// V1" dot V2^ - V1^ dot V2"
    r2 = _mm_sub_ps(_mm_ror_ps(Vb,2),_mm_ror_ps(Vb,0));		// V1^ dot V2' - V1' dot V2^
    r3 = _mm_sub_ps(_mm_ror_ps(Va,0),_mm_ror_ps(Vc,1));		// V1' dot V2" - V1" dot V2'

    tt = _mm_ror_ps(_L4,1);		sum = _mm_mul_ps(tt,r1);
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r2));
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r3));
    __m128 mtL3 = _mm_xor_ps(sum,Sign_PNPN);

    // Dividing is FASTER than rcp_nr! (Because rcp_nr causes many register-memory RWs).
    MCSF_3D_ALIGN16 const float _vmathZERONE[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    RDet = _mm_div_ss(_mm_load_ss((float *)&_vmathZERONE), Det); // TODO: just 1.0f?
    RDet = _mm_shuffle_ps(RDet,RDet,0x00);

    // Devide the first 12 minterms with the determinant.
    mtL1 = _mm_mul_ps(mtL1, RDet);
    mtL2 = _mm_mul_ps(mtL2, RDet);
    mtL3 = _mm_mul_ps(mtL3, RDet);

    // Calculate the minterms of the forth line and devide by the determinant.
    tt = _mm_ror_ps(_L3,1);		sum = _mm_mul_ps(tt,r1);
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r2));
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r3));
    __m128 mtL4 = _mm_xor_ps(sum,Sign_NPNP);
    mtL4 = _mm_mul_ps(mtL4, RDet);

    // Now we just have to transpose the minterms matrix.
    trns0 = _mm_unpacklo_ps(mtL1,mtL2);
    trns1 = _mm_unpacklo_ps(mtL3,mtL4);
    trns2 = _mm_unpackhi_ps(mtL1,mtL2);
    trns3 = _mm_unpackhi_ps(mtL3,mtL4);
    _L1 = _mm_movelh_ps(trns0,trns1);
    _L2 = _mm_movehl_ps(trns1,trns0);
    _L3 = _mm_movelh_ps(trns2,trns3);
    _L4 = _mm_movehl_ps(trns3,trns2);

    return Matrix4f(
        Vector4f( _L1 ),
        Vector4f( _L2 ),
        Vector4f( _L3 ),
        Vector4f( _L4 )
        );
}

 
    const Matrix4f AffineInverse( const Matrix4f & mat )
{
    Vector3f col0 = mat.GetCol0().GetXYZ(); 
    Vector3f col1 = mat.GetCol1().GetXYZ();
    Vector3f col2 = mat.GetCol2().GetXYZ();
    Vector3f col3 = mat.GetCol3().GetXYZ();

    Vector3f tmp0, tmp1, tmp2, inv0, inv1, inv2;

    tmp0 = CrossProduct( col1, col2 );
    tmp1 = CrossProduct( col2, col0 );
    tmp2 = CrossProduct( col0, col1 );
    float detinv = ( 1.0f / DotProduct( col2, tmp2 ) );

    inv0 = Vector3f( ( tmp0.GetX() * detinv ), ( tmp1.GetX() * detinv ), ( tmp2.GetX() * detinv ) );
    inv1 = Vector3f( ( tmp0.GetY() * detinv ), ( tmp1.GetY() * detinv ), ( tmp2.GetY() * detinv ) );
    inv2 = Vector3f( ( tmp0.GetZ() * detinv ), ( tmp1.GetZ() * detinv ), ( tmp2.GetZ() * detinv ) );

    Vector3f temp = Vector3f( 
        -( ( inv0 * col3.GetX() ) + ( ( inv1 * col3.GetY() ) + ( inv2 * col3.GetZ() ) ) ) 
        );

    return Matrix4f(
        Vector4f(inv0, 0.0f),
        Vector4f(inv1, 0.0f),
        Vector4f(inv2, 0.0f),
        Vector4f(temp, 1.0)
        );
}


 
    const Matrix4f OrthoInverse( const Matrix4f & mat )
{
    Vector3f inv0 = mat.GetRow(0).GetXYZ();
    Vector3f inv1 = mat.GetRow(1).GetXYZ();
    Vector3f inv2 = mat.GetRow(2).GetXYZ();
    Vector3f temp = Vector3f( 
        -( ( inv0 * mat.GetCol3().GetX() ) + ( ( inv1 * mat.GetCol3().GetY() ) + ( inv2 * mat.GetCol3().GetZ() ) ) )
        );
    return Matrix4f(
        Vector4f(inv0, 0.0f),
        Vector4f(inv1, 0.0f),
        Vector4f(inv2, 0.0f),
        Vector4f(temp, 1.0)
        );
}

 
    const float Determinant( const Matrix4f & mat )
{
    __m128 Va,Vb,Vc;
    __m128 r1,r2,r3,tt,tt2;
    __m128 sum,Det;

    __m128 _L1 = mat.GetCol0().m_Vec128;
    __m128 _L2 = mat.GetCol1().m_Vec128;
    __m128 _L3 = mat.GetCol2().m_Vec128;
    __m128 _L4 = mat.GetCol3().m_Vec128;
    // Calculating the minterms for the first line.

    // _mm_ror_ps is just a macro using _mm_shuffle_ps().
    tt = _L4; tt2 = _mm_ror_ps(_L3,1); 
    Vc = _mm_mul_ps(tt2,_mm_ror_ps(tt,0));					// V3' dot V4
    Va = _mm_mul_ps(tt2,_mm_ror_ps(tt,2));					// V3' dot V4"
    Vb = _mm_mul_ps(tt2,_mm_ror_ps(tt,3));					// V3' dot V4^

    r1 = _mm_sub_ps(_mm_ror_ps(Va,1),_mm_ror_ps(Vc,2));		// V3" dot V4^ - V3^ dot V4"
    r2 = _mm_sub_ps(_mm_ror_ps(Vb,2),_mm_ror_ps(Vb,0));		// V3^ dot V4' - V3' dot V4^
    r3 = _mm_sub_ps(_mm_ror_ps(Va,0),_mm_ror_ps(Vc,1));		// V3' dot V4" - V3" dot V4'

    tt = _L2;
    Va = _mm_ror_ps(tt,1);		sum = _mm_mul_ps(Va,r1);
    Vb = _mm_ror_ps(tt,2);		sum = _mm_add_ps(sum,_mm_mul_ps(Vb,r2));
    Vc = _mm_ror_ps(tt,3);		sum = _mm_add_ps(sum,_mm_mul_ps(Vc,r3));

    // Calculating the determinant.
    Det = _mm_mul_ps(sum,_L1);
    Det = _mm_add_ps(Det,_mm_movehl_ps(Det,Det));

    // Calculating the minterms of the second line (using previous results).
    tt = _mm_ror_ps(_L1,1);		sum = _mm_mul_ps(tt,r1);
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r2));
    tt = _mm_ror_ps(tt,1);		sum = _mm_add_ps(sum,_mm_mul_ps(tt,r3));

    // Testing the determinant.
    Det = _mm_sub_ss(Det,_mm_shuffle_ps(Det,Det,1));
    return _vmathVfGetElement(Det, 0);
}

#ifdef _DEBUG

 
    void Print( const Matrix4f & mat )
{
    print( mat.GetRow( 0 ) );
    print( mat.GetRow( 1 ) );
    print( mat.GetRow( 2 ) );
    print( mat.GetRow( 3 ) );
}

 
    void Print( const Matrix4f & mat, const char * name )
{
    printf("%s:\n", name);
    Print( mat );
}

#endif


MED_IMAGING_END_NAMESPACE