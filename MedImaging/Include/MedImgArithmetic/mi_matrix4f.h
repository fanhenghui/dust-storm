#ifndef MED_IMAGING_ARITHMETIC_MATRIX4X4_H_
#define MED_IMAGING_ARITHMETIC_MATRIX4X4_H_

#include "MedImgArithmetic/mi_arithmetic_common.h"
#include "MedImgArithmetic/mi_vector4f.h"

MED_IMAGING_BEGIN_NAMESPACE

/// \class Matrix4f
/// 
/// \brief *****
class Arithmetic_Export Matrix4f
{
    Vector4f m_Col0;
    Vector4f m_Col1;
    Vector4f m_Col2;
    Vector4f m_Col3;

public:
    Matrix4f()
    {
        m_Col0 = Vector4f(0.0f);
        m_Col1 = Vector4f(0.0f);
        m_Col2 = Vector4f(0.0f);
        m_Col3 = Vector4f(0.0f);
    }

    Matrix4f( const Matrix4f & mat )
    {
        m_Col0 = mat.m_Col0;
        m_Col1 = mat.m_Col1;
        m_Col2 = mat.m_Col2;
        m_Col3 = mat.m_Col3;
    }

    Matrix4f( const Vector4f &col0, const Vector4f &col1, const Vector4f &col2, const Vector4f &col3 )
    {
        m_Col0 = col0;
        m_Col1 = col1;
        m_Col2 = col2;
        m_Col3 = col3;
    }

    Matrix4f( const float scalar )
    {
        m_Col0 = Vector4f( scalar );
        m_Col1 = Vector4f( scalar );
        m_Col2 = Vector4f( scalar );
        m_Col3 = Vector4f( scalar );
    }

    inline 
        Matrix4f & operator=( const Matrix4f & mat )
    {
        m_Col0 = mat.m_Col0;
        m_Col1 = mat.m_Col1;
        m_Col2 = mat.m_Col2;
        m_Col3 = mat.m_Col3;
        return *this;
    }

    inline 
        Matrix4f & SetTranslation( const Vector3f &translateVec )
    {
        m_Col3.SetXYZ( translateVec );
        return *this;
    }

    inline 
        const Vector3f GetTranslation() const
    {
        return m_Col3.GetXYZ( );
    }

    inline 
        Matrix4f & SetCol0( const Vector4f &col0 )
    {
        m_Col0 = col0;
        return *this;
    }

    inline 
        Matrix4f & SetCol1( const Vector4f &col1 )
    {
        m_Col1 = col1;
        return *this;
    }

    inline 
        Matrix4f & SetCol2( const Vector4f &col2 )
    {
        m_Col2 = col2;
        return *this;
    }

    inline 
        Matrix4f & SetCol3( const Vector4f &col3 )
    {
        m_Col3 = col3;
        return *this;
    }

    inline 
        const Vector4f GetCol0() const
    {
        return m_Col0;
    }

    inline 
        const Vector4f GetCol1() const
    {
        return m_Col1;
    }

    inline 
        const Vector4f GetCol2() const
    {
        return m_Col2;
    }

    inline 
        const Vector4f GetCol3() const
    {
        return m_Col3;
    }

    inline 
        Matrix4f & SetCol( const int col, const Vector4f &vec )
    {
        *(&m_Col0 + col) = vec;
        return *this;
    }

    inline 
        const Vector4f GetCol( const int col ) const
    {
        return *(&m_Col0 + col);
    }

    inline 
        Matrix4f & SetRow( const int row, const Vector4f &vec )
    {
        m_Col0.SetElem( row, vec.GetElem( 0 ) );
        m_Col1.SetElem( row, vec.GetElem( 1 ) );
        m_Col2.SetElem( row, vec.GetElem( 2 ) );
        m_Col3.SetElem( row, vec.GetElem( 3 ) );
        return *this;
    }

    inline 
        const Vector4f GetRow( const int row ) const
    {
        return Vector4f( m_Col0.GetElem( row ), m_Col1.GetElem( row ), m_Col2.GetElem( row ), m_Col3.GetElem( row ) );
    }

    inline 
        Vector4f & operator[]( const int col )
    {
        return *(&m_Col0 + col);
    }

    inline 
        const Vector4f operator[]( const int col ) const
    {
        return *(&m_Col0 + col);
    }

    inline 
        Matrix4f & SetElem( const int col, const int row, const float val )
    {
        Vector4f tmpV3_0;
        tmpV3_0 = this->GetCol( col );
        tmpV3_0.SetElem( row, val );
        this->SetCol( col, tmpV3_0 );
        return *this;
    }

    inline 
        const float GetElem( const int col, const int row ) const
    {
        return this->GetCol( col ).GetElem( row );
    }

    inline 
        const Matrix4f operator+( const Matrix4f & mat ) const
    {
        return Matrix4f(
            ( m_Col0 + mat.m_Col0 ),
            ( m_Col1 + mat.m_Col1 ),
            ( m_Col2 + mat.m_Col2 ),
            ( m_Col3 + mat.m_Col3 )
            );
    }

    inline 
        const Matrix4f operator-( const Matrix4f & mat ) const
    {
        return Matrix4f(
            ( m_Col0 - mat.m_Col0 ),
            ( m_Col1 - mat.m_Col1 ),
            ( m_Col2 - mat.m_Col2 ),
            ( m_Col3 - mat.m_Col3 )
            );
    }

    inline 
        const Matrix4f operator-() const
    {
        return Matrix4f(
            ( -m_Col0 ),
            ( -m_Col1 ),
            ( -m_Col2 ),
            ( -m_Col3 )
            );
    }

    inline 
        const Matrix4f operator*( const float scalar ) const
    {
        return Matrix4f(
            ( m_Col0 * scalar ),
            ( m_Col1 * scalar ),
            ( m_Col2 * scalar ),
            ( m_Col3 * scalar )
            );
    }

    inline const Vector4f operator*( const Vector4f &vec ) const
    {
        return Vector4f(
            _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(m_Col0.m_Vec128, _mm_shuffle_ps(vec.m_Vec128, vec.m_Vec128, _MM_SHUFFLE(0,0,0,0))), 
            _mm_mul_ps(m_Col1.m_Vec128, _mm_shuffle_ps(vec.m_Vec128, vec.m_Vec128, _MM_SHUFFLE(1,1,1,1)))),
            _mm_add_ps(_mm_mul_ps(m_Col2.m_Vec128, _mm_shuffle_ps(vec.m_Vec128, vec.m_Vec128, _MM_SHUFFLE(2,2,2,2))), 
            _mm_mul_ps(m_Col3.m_Vec128, _mm_shuffle_ps(vec.m_Vec128, vec.m_Vec128, _MM_SHUFFLE(3,3,3,3)))))
            );
    }

    inline 
        const Matrix4f operator*( const Matrix4f & mat ) const
    {
        return Matrix4f(
            ( *this * mat.m_Col0 ),
            ( *this * mat.m_Col1 ),
            ( *this * mat.m_Col2 ),
            ( *this * mat.m_Col3 )
            );
    }

    inline
        Matrix4f & operator+=( const Matrix4f & mat )
    {
        *this = *this + mat;
        return *this;
    }

    inline 
        Matrix4f & operator-=( const Matrix4f & mat )
    {
        *this = *this - mat;
        return *this;
    }

    inline 
        Matrix4f & operator*=( const float scalar )
    {
        *this = *this * scalar;
        return *this;
    }

    inline 
        Matrix4f & operator*=( const Matrix4f & mat )
    {
        *this = *this * mat;
        return *this;
    }

    inline 
        const Matrix4f Identity()
    {
        return Matrix4f(
            Vector4f(_mm_setr_ps(1.0f,0.0f,0.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,1.0f,0.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,0.0f,1.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f RotationX( const float radians )
    {
        __m128 s, c, res1, res2;
        __m128 zero;
        MCSF_3D_ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        zero = _mm_setzero_ps();
        sincosf4( _mm_set1_ps(radians), &s, &c );
        res1 = vec_sel( zero, c, select_y );
        res1 = vec_sel( res1, s, select_z );
        res2 = vec_sel( zero, negatef4(s), select_y );
        res2 = vec_sel( res2, c, select_z );
        return Matrix4f(
            Vector4f(_mm_setr_ps(1.0f,0.0f,0.0f,0.0f)),
            Vector4f( res1 ),
            Vector4f( res2 ),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f RotationY( const float radians )
    {
        __m128 s, c, res0, res2;
        __m128 zero;
        MCSF_3D_ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        zero = _mm_setzero_ps();
        sincosf4( _mm_set1_ps(radians), &s, &c );
        res0 = vec_sel( zero, c, select_x );
        res0 = vec_sel( res0, negatef4(s), select_z );
        res2 = vec_sel( zero, s, select_x );
        res2 = vec_sel( res2, c, select_z );
        return Matrix4f(
            Vector4f( res0 ),
            Vector4f(_mm_setr_ps(0.0f,1.0f,0.0f,0.0f)),
            Vector4f( res2 ),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f RotationZ( const float radians )
    {
        __m128 s, c, res0, res1;
        __m128 zero;
        MCSF_3D_ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        zero = _mm_setzero_ps();
        sincosf4( _mm_set1_ps(radians), &s, &c );
        res0 = vec_sel( zero, c, select_x );
        res0 = vec_sel( res0, s, select_y );
        res1 = vec_sel( zero, negatef4(s), select_x );
        res1 = vec_sel( res1, c, select_y );
        return Matrix4f(
            Vector4f( res0 ),
            Vector4f( res1 ),
            Vector4f(_mm_setr_ps(0.0f,0.0f,1.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f RotationZYX( const Vector3f &radiansXYZ )
    {
        __m128 angles, s, negS, c, X0, X1, Y0, Y1, Z0, Z1, tmp;
        angles = Vector4f( radiansXYZ, 0.0f ).m_Vec128;
        sincosf4( angles, &s, &c );
        negS = negatef4( s );
        Z0 = vec_mergel( c, s );
        Z1 = vec_mergel( negS, c );
        MCSF_3D_ALIGN16 unsigned int select_xyz[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0};
        Z1 = vec_and( Z1, _mm_load_ps( (float *)select_xyz ) );
        Y0 = _mm_shuffle_ps( c, negS, _MM_SHUFFLE(0,1,1,1) );
        Y1 = _mm_shuffle_ps( s, c, _MM_SHUFFLE(0,1,1,1) );
        X0 = vec_splat( s, 0 );
        X1 = vec_splat( c, 0 );
        tmp = vec_mul( Z0, Y1 );
        return Matrix4f(
            Vector4f( vec_mul( Z0, Y0 ) ),
            Vector4f( vec_madd( Z1, X1, vec_mul( tmp, X0 ) ) ),
            Vector4f( vec_nmsub( Z1, X0, vec_mul( tmp, X1 ) ) ),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f Rotation( const float radians, const Vector3f &unitVec )
    {
        __m128 axis, s, c, oneMinusC, axisS, negAxisS, xxxx, yyyy, zzzz, tmp0, tmp1, tmp2;
        axis = unitVec.m_Vec128;
        sincosf4( _mm_set1_ps(radians), &s, &c );
        xxxx = vec_splat( axis, 0 );
        yyyy = vec_splat( axis, 1 );
        zzzz = vec_splat( axis, 2 );
        oneMinusC = vec_sub( _mm_set1_ps(1.0f), c );
        axisS = vec_mul( axis, s );
        negAxisS = negatef4( axisS );
        MCSF_3D_ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        //tmp0 = vec_perm( axisS, negAxisS, _VECTORMATH_PERM_XZBX );
        tmp0 = _mm_shuffle_ps( axisS, axisS, _MM_SHUFFLE(0,0,2,0) );
        tmp0 = vec_sel(tmp0, vec_splat(negAxisS, 1), select_z);
        //tmp1 = vec_perm( axisS, negAxisS, _VECTORMATH_PERM_CXXX );
        tmp1 = vec_sel( vec_splat(axisS, 0), vec_splat(negAxisS, 2), select_x );
        //tmp2 = vec_perm( axisS, negAxisS, _VECTORMATH_PERM_YAXX );
        tmp2 = _mm_shuffle_ps( axisS, axisS, _MM_SHUFFLE(0,0,0,1) );
        tmp2 = vec_sel(tmp2, vec_splat(negAxisS, 0), select_y);
        tmp0 = vec_sel( tmp0, c, select_x );
        tmp1 = vec_sel( tmp1, c, select_y );
        tmp2 = vec_sel( tmp2, c, select_z );
        MCSF_3D_ALIGN16 unsigned int select_xyz[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0};
        axis = vec_and( axis, _mm_load_ps( (float *)select_xyz ) );
        tmp0 = vec_and( tmp0, _mm_load_ps( (float *)select_xyz ) );
        tmp1 = vec_and( tmp1, _mm_load_ps( (float *)select_xyz ) );
        tmp2 = vec_and( tmp2, _mm_load_ps( (float *)select_xyz ) );
        return Matrix4f(
            Vector4f( vec_madd( vec_mul( axis, xxxx ), oneMinusC, tmp0 ) ),
            Vector4f( vec_madd( vec_mul( axis, yyyy ), oneMinusC, tmp1 ) ),
            Vector4f( vec_madd( vec_mul( axis, zzzz ), oneMinusC, tmp2 ) ),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f Scale( const Vector3f &scaleVec )
    {
        __m128 zero = _mm_setzero_ps();
        MCSF_3D_ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        MCSF_3D_ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        return Matrix4f(
            Vector4f( vec_sel( zero, scaleVec.m_Vec128, select_x ) ),
            Vector4f( vec_sel( zero, scaleVec.m_Vec128, select_y ) ),
            Vector4f( vec_sel( zero, scaleVec.m_Vec128, select_z ) ),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        const Matrix4f Translation( const Vector3f &translateVec )
    {
        return Matrix4f(
            Vector4f(_mm_setr_ps(1.0f,0.0f,0.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,1.0f,0.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,0.0f,1.0f,0.0f)),
            Vector4f( translateVec, 1.0f )
            );
    }

    inline 
        void SetIdentity()
    {
        *this = Matrix4f(
            Vector4f(_mm_setr_ps(1.0f,0.0f,0.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,1.0f,0.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,0.0f,1.0f,0.0f)),
            Vector4f(_mm_setr_ps(0.0f,0.0f,0.0f,1.0f))
            );
    }

    inline 
        void Prepend( const Matrix4f &myMatrix )
    {
        *this = myMatrix * (*this);
    }

    inline 
        void Append( const Matrix4f &myMatrix )
    {
        *this = (*this) * myMatrix;
    }

    inline 
        void Transpose()
    {
        __m128 tmp0, tmp1, tmp2, tmp3, res0, res1, res2, res3;
        tmp0 = vec_mergeh( m_Col0.m_Vec128, m_Col2.m_Vec128 );
        tmp1 = vec_mergeh( m_Col1.m_Vec128, m_Col3.m_Vec128 );
        tmp2 = vec_mergel( m_Col0.m_Vec128, m_Col2.m_Vec128 );
        tmp3 = vec_mergel( m_Col1.m_Vec128, m_Col3.m_Vec128 );
        res0 = vec_mergeh( tmp0, tmp1 );
        res1 = vec_mergel( tmp0, tmp1 );
        res2 = vec_mergeh( tmp2, tmp3 );
        res3 = vec_mergel( tmp2, tmp3 );
        *this =  Matrix4f(
            Vector4f( res0 ),
            Vector4f( res1 ),
            Vector4f( res2 ),
            Vector4f( res3 )
            );
    }

    inline 
        bool HasInverse() const
    {
        return (fabs(Determinant()) > FLOAT_EPSILON);
    }

    inline 
        Matrix4f Inverse() const
    {
        __m128 Va,Vb,Vc;
        __m128 r1,r2,r3,tt,tt2;
        __m128 sum,Det,RDet;
        __m128 trns0,trns1,trns2,trns3;

        __m128 _L1 = m_Col0.m_Vec128;
        __m128 _L2 = m_Col1.m_Vec128;
        __m128 _L3 = m_Col2.m_Vec128;
        __m128 _L4 = m_Col3.m_Vec128;
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

    inline 
        Vector3f TransformPoint( const Vector3f& mypoint ) const
    {
        Vector4f spoint = Vector4f(mypoint);
        spoint._m[3] = 1.0f;
        Vector4f temp = *this * spoint;
        float dw = temp.GetElem(3);
        dw = fabs(dw) > FLOAT_EPSILON ? dw : FLOAT_EPSILON;
        dw = 1.0f/dw;
        temp = temp * dw;
        return Vector3f(temp.m_Vec128);
    }

    inline 
        Vector3f TransformVector( const Vector3f& vec ) const
    {
        __m128 res;
        __m128 xxxx, yyyy, zzzz;
        xxxx = vec_splat( vec.m_Vec128, 0 );
        yyyy = vec_splat( vec.m_Vec128, 1 );
        zzzz = vec_splat( vec.m_Vec128, 2 );
        res = vec_mul( m_Col0.m_Vec128, xxxx );
        res = vec_madd( m_Col1.m_Vec128, yyyy, res );
        res = vec_madd( m_Col2.m_Vec128, zzzz, res );
        return Vector3f( res );
    }

    inline 
        void ExtractTranslate( Vector3f &myVector) const
    {
        myVector = Vector3f(m_Col3.m_Vec128);
    }

    inline 
        float Determinant() const
    {
        __m128 Va,Vb,Vc;
        __m128 r1,r2,r3,tt,tt2;
        __m128 sum,Det;

        __m128 _L1 = m_Col0.m_Vec128;
        __m128 _L2 = m_Col1.m_Vec128;
        __m128 _L3 = m_Col2.m_Vec128;
        __m128 _L4 = m_Col3.m_Vec128;
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

    inline 
        Matrix4f GetTranspose() const
    {
        Matrix4f tmp = *this;
        tmp.Transpose();
        return tmp;
    }

    inline 
        bool operator==( const Matrix4f& myMatrix ) const
    {
        return (m_Col0 == myMatrix.GetCol0() && 
            m_Col1 == myMatrix.GetCol1() &&
            m_Col2 == myMatrix.GetCol2() &&
            m_Col3 == myMatrix.GetCol3() );
    }

    inline 
        bool operator!=( const Matrix4f& myMatrix ) const
    {
        return (m_Col0 != myMatrix.GetCol0() || 
            m_Col1 != myMatrix.GetCol1() ||
            m_Col2 != myMatrix.GetCol2() ||
            m_Col3 != myMatrix.GetCol3() );
    }

    inline 
        bool IsAffine() const
    {
        Vector4f tmp = this->GetRow(3);
        return (tmp == Vector4f(FLOAT_EPSILON, FLOAT_EPSILON, FLOAT_EPSILON, FLOAT_EPSILON+1.0f));
    }

    inline 
        Matrix4f MakeScale( const Vector3f& myScale )
    {
        SetIdentity();
        m_Col0 *= myScale.GetX();
        m_Col1 *= myScale.GetY();
        m_Col2 *= myScale.GetZ();
        return *this;
    }

    inline 
        Matrix4f MakeTranslate( const Vector3f& myTranslate )
    {
        SetIdentity();
        m_Col3 += Vector4f(myTranslate, 0.0f);
        return *this;
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// Append (post-multiply) a scale transformation to a 4x4 matrix
// Faster than creating and multiplying a scale transformation matrix.
Arithmetic_Export inline 
    const Matrix4f AppendScale( const Matrix4f & mat, const Vector3f &scaleVec );

// Prepend (pre-multiply) a scale transformation to a 4x4 matrix
// Faster than creating and multiplying a scale transformation matrix.
Arithmetic_Export inline 
    const Matrix4f PrependScale( const Vector3f &scaleVec, const Matrix4f & mat );

// Multiply two 4x4 matrices per element
Arithmetic_Export inline 
    const Matrix4f MulPerElem( const Matrix4f & mat0, const Matrix4f & mat1 );

// Compute the absolute value of a 4x4 matrix per element
Arithmetic_Export inline 
    const Matrix4f AbsPerElem( const Matrix4f & mat );

// Transpose of a 4x4 matrix
Arithmetic_Export inline 
    const Matrix4f Transpose( const Matrix4f & mat );

// Compute the inverse of a 4x4 matrix
// Result is unpredictable when the determinant of mat is equal to or near 0.
Arithmetic_Export inline 
    const Matrix4f Inverse( const Matrix4f & mat );

// Compute the inverse of a 4x4 matrix, which is expected to be an affine matrix
// This can be used to achieve better performance than a general inverse when the specified 4x4 matrix meets the given restrictions.  The result is unpredictable when the determinant of mat is equal to or near 0.
Arithmetic_Export inline 
    const Matrix4f AffineInverse( const Matrix4f & mat );

// Compute the inverse of a 4x4 matrix, which is expected to be an affine matrix with an orthogonal upper-left 3x3 submatrix
// This can be used to achieve better performance than a general inverse when the specified 4x4 matrix meets the given restrictions.
Arithmetic_Export inline 
    const Matrix4f OrthoInverse( const Matrix4f & mat );

// Determinant of a 4x4 matrix
Arithmetic_Export inline 
    const float Determinant( const Matrix4f & mat );

#ifdef _DEBUG

// Print a 4x4 matrix
// Function is only defined when _DEBUG is defined.
Arithmetic_Export inline 
    void Print( const Matrix4f & mat );

// Print a 4x4 matrix and an associated string identifier
// Function is only defined when _DEBUG is defined.
Arithmetic_Export inline 
    void Print( const Matrix4f & mat, const char * name );

#endif

MED_IMAGING_END_NAMESPACE

#endif