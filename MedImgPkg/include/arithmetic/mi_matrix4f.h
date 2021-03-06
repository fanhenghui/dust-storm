#ifndef MEDIMGARITHMETIC_MI_MATRIX4F_H
#define MEDIMGARITHMETIC_MI_MATRIX4F_H

#include "arithmetic/mi_simd.h"
#include "arithmetic/mi_vector4f.h"

MED_IMG_BEGIN_NAMESPACE

/// \class Matrix4f
///
/// \brief *****
class Arithmetic_Export Matrix4f {
    Vector4f m_Col0;
    Vector4f m_Col1;
    Vector4f m_Col2;
    Vector4f m_Col3;

public:
    Matrix4f() {
        m_Col0 = Vector4f(0.0f);
        m_Col1 = Vector4f(0.0f);
        m_Col2 = Vector4f(0.0f);
        m_Col3 = Vector4f(0.0f);
    }

    Matrix4f(const Matrix4f& mat) {
        m_Col0 = mat.m_Col0;
        m_Col1 = mat.m_Col1;
        m_Col2 = mat.m_Col2;
        m_Col3 = mat.m_Col3;
    }

    Matrix4f(const Vector4f& col0, const Vector4f& col1, const Vector4f& col2,
             const Vector4f& col3) {
        m_Col0 = col0;
        m_Col1 = col1;
        m_Col2 = col2;
        m_Col3 = col3;
    }

    Matrix4f(const float scalar) {
        m_Col0 = Vector4f(scalar);
        m_Col1 = Vector4f(scalar);
        m_Col2 = Vector4f(scalar);
        m_Col3 = Vector4f(scalar);
    }

    inline Matrix4f& operator=(const Matrix4f& mat) {
        m_Col0 = mat.m_Col0;
        m_Col1 = mat.m_Col1;
        m_Col2 = mat.m_Col2;
        m_Col3 = mat.m_Col3;
        return *this;
    }

    inline Matrix4f& set_translation(const Vector3f& translateVec) {
        m_Col3.set_xyz(translateVec);
        return *this;
    }

    inline const Vector3f get_translation() const {
        return m_Col3.get_xyz();
    }

    inline Matrix4f& set_col0(const Vector4f& col0) {
        m_Col0 = col0;
        return *this;
    }

    inline Matrix4f& set_col1(const Vector4f& col1) {
        m_Col1 = col1;
        return *this;
    }

    inline Matrix4f& set_col2(const Vector4f& col2) {
        m_Col2 = col2;
        return *this;
    }

    inline Matrix4f& set_col3(const Vector4f& col3) {
        m_Col3 = col3;
        return *this;
    }

    inline const Vector4f get_col0() const {
        return m_Col0;
    }

    inline const Vector4f get_col1() const {
        return m_Col1;
    }

    inline const Vector4f get_col2() const {
        return m_Col2;
    }

    inline const Vector4f get_col3() const {
        return m_Col3;
    }

    inline Matrix4f& set_col(const int col, const Vector4f& vec) {
        *(&m_Col0 + col) = vec;
        return *this;
    }

    inline const Vector4f get_col(const int col) const {
        return *(&m_Col0 + col);
    }

    inline Matrix4f& set_row(const int row, const Vector4f& vec) {
        m_Col0.set_elem(row, vec.get_elem(0));
        m_Col1.set_elem(row, vec.get_elem(1));
        m_Col2.set_elem(row, vec.get_elem(2));
        m_Col3.set_elem(row, vec.get_elem(3));
        return *this;
    }

    inline const Vector4f get_row(const int row) const {
        return Vector4f(m_Col0.get_elem(row), m_Col1.get_elem(row),
                        m_Col2.get_elem(row), m_Col3.get_elem(row));
    }

    inline Vector4f& operator[](const int col) {
        return *(&m_Col0 + col);
    }

    inline const Vector4f operator[](const int col) const {
        return *(&m_Col0 + col);
    }

    inline Matrix4f& set_elem(const int col, const int row, const float val) {
        Vector4f tmpV3_0;
        tmpV3_0 = this->get_col(col);
        tmpV3_0.set_elem(row, val);
        this->set_col(col, tmpV3_0);
        return *this;
    }

    inline const float get_elem(const int col, const int row) const {
        return this->get_col(col).get_elem(row);
    }

    inline const Matrix4f operator+(const Matrix4f& mat) const {
        return Matrix4f((m_Col0 + mat.m_Col0), (m_Col1 + mat.m_Col1),
                        (m_Col2 + mat.m_Col2), (m_Col3 + mat.m_Col3));
    }

    inline const Matrix4f operator-(const Matrix4f& mat) const {
        return Matrix4f((m_Col0 - mat.m_Col0), (m_Col1 - mat.m_Col1),
                        (m_Col2 - mat.m_Col2), (m_Col3 - mat.m_Col3));
    }

    inline const Matrix4f operator-() const {
        return Matrix4f((-m_Col0), (-m_Col1), (-m_Col2), (-m_Col3));
    }

    inline const Matrix4f operator*(const float scalar) const {
        return Matrix4f((m_Col0 * scalar), (m_Col1 * scalar), (m_Col2 * scalar),
                        (m_Col3 * scalar));
    }

    inline const Vector4f operator*(const Vector4f& vec) const {
        return Vector4f(_mm_add_ps(
                            _mm_add_ps(
                                _mm_mul_ps(m_Col0._m128, _mm_shuffle_ps(vec._m128, vec._m128,
                                           _MM_SHUFFLE(0, 0, 0, 0))),
                                _mm_mul_ps(m_Col1._m128, _mm_shuffle_ps(vec._m128, vec._m128,
                                           _MM_SHUFFLE(1, 1, 1, 1)))),
                            _mm_add_ps(
                                _mm_mul_ps(m_Col2._m128, _mm_shuffle_ps(vec._m128, vec._m128,
                                           _MM_SHUFFLE(2, 2, 2, 2))),
                                _mm_mul_ps(m_Col3._m128,
                                           _mm_shuffle_ps(vec._m128, vec._m128,
                                                   _MM_SHUFFLE(3, 3, 3, 3))))));
    }

    inline const Matrix4f operator*(const Matrix4f& mat) const {
        return Matrix4f((*this * mat.m_Col0), (*this * mat.m_Col1),
                        (*this * mat.m_Col2), (*this * mat.m_Col3));
    }

    inline Matrix4f& operator+=(const Matrix4f& mat) {
        *this = *this + mat;
        return *this;
    }

    inline Matrix4f& operator-=(const Matrix4f& mat) {
        *this = *this - mat;
        return *this;
    }

    inline Matrix4f& operator*=(const float scalar) {
        *this = *this * scalar;
        return *this;
    }

    inline Matrix4f& operator*=(const Matrix4f& mat) {
        *this = *this * mat;
        return *this;
    }

    inline const Matrix4f identity() {
        return Matrix4f(Vector4f(_mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f)),
                        Vector4f(_mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f)),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline const Matrix4f rotation_x(const float radians) {
        __m128 s, c, res1, res2;
        __m128 zero;
        ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        zero = _mm_setzero_ps();
        sincosf4(_mm_set1_ps(radians), &s, &c);
        res1 = vec_sel(zero, c, select_y);
        res1 = vec_sel(res1, s, select_z);
        res2 = vec_sel(zero, negatef4(s), select_y);
        res2 = vec_sel(res2, c, select_z);
        return Matrix4f(Vector4f(_mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f)),
                        Vector4f(res1), Vector4f(res2),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline const Matrix4f rotation_y(const float radians) {
        __m128 s, c, res0, res2;
        __m128 zero;
        ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        zero = _mm_setzero_ps();
        sincosf4(_mm_set1_ps(radians), &s, &c);
        res0 = vec_sel(zero, c, select_x);
        res0 = vec_sel(res0, negatef4(s), select_z);
        res2 = vec_sel(zero, s, select_x);
        res2 = vec_sel(res2, c, select_z);
        return Matrix4f(
                   Vector4f(res0), Vector4f(_mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)),
                   Vector4f(res2), Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline const Matrix4f rotation_z(const float radians) {
        __m128 s, c, res0, res1;
        __m128 zero;
        ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        zero = _mm_setzero_ps();
        sincosf4(_mm_set1_ps(radians), &s, &c);
        res0 = vec_sel(zero, c, select_x);
        res0 = vec_sel(res0, s, select_y);
        res1 = vec_sel(zero, negatef4(s), select_x);
        res1 = vec_sel(res1, c, select_y);
        return Matrix4f(Vector4f(res0), Vector4f(res1),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f)),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline const Matrix4f rotate(const float radians, const Vector3f& unitVec) {
        __m128 axis, s, c, oneMinusC, axisS, negAxisS, xxxx, yyyy, zzzz, tmp0, tmp1,
               tmp2;
        axis = unitVec._m128;
        sincosf4(_mm_set1_ps(radians), &s, &c);
        xxxx = vec_splat(axis, 0);
        yyyy = vec_splat(axis, 1);
        zzzz = vec_splat(axis, 2);
        oneMinusC = vec_sub(_mm_set1_ps(1.0f), c);
        axisS = vec_mul(axis, s);
        negAxisS = negatef4(axisS);
        ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        // tmp0 = vec_perm( axisS, negAxisS, _VECTORMATH_PERM_XZBX );
        tmp0 = _mm_shuffle_ps(axisS, axisS, _MM_SHUFFLE(0, 0, 2, 0));
        tmp0 = vec_sel(tmp0, vec_splat(negAxisS, 1), select_z);
        // tmp1 = vec_perm( axisS, negAxisS, _VECTORMATH_PERM_CXXX );
        tmp1 = vec_sel(vec_splat(axisS, 0), vec_splat(negAxisS, 2), select_x);
        // tmp2 = vec_perm( axisS, negAxisS, _VECTORMATH_PERM_YAXX );
        tmp2 = _mm_shuffle_ps(axisS, axisS, _MM_SHUFFLE(0, 0, 0, 1));
        tmp2 = vec_sel(tmp2, vec_splat(negAxisS, 0), select_y);
        tmp0 = vec_sel(tmp0, c, select_x);
        tmp1 = vec_sel(tmp1, c, select_y);
        tmp2 = vec_sel(tmp2, c, select_z);
        ALIGN16 unsigned int select_xyz[4] = {0xffffffff, 0xffffffff, 0xffffffff,
                                              0
                                             };
        axis = vec_and(axis, _mm_load_ps((float*)select_xyz));
        tmp0 = vec_and(tmp0, _mm_load_ps((float*)select_xyz));
        tmp1 = vec_and(tmp1, _mm_load_ps((float*)select_xyz));
        tmp2 = vec_and(tmp2, _mm_load_ps((float*)select_xyz));
        return Matrix4f(Vector4f(vec_madd(vec_mul(axis, xxxx), oneMinusC, tmp0)),
                        Vector4f(vec_madd(vec_mul(axis, yyyy), oneMinusC, tmp1)),
                        Vector4f(vec_madd(vec_mul(axis, zzzz), oneMinusC, tmp2)),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline const Matrix4f scale(const Vector3f& scaleVec) {
        __m128 zero = _mm_setzero_ps();
        ALIGN16 unsigned int select_x[4] = {0xffffffff, 0, 0, 0};
        ALIGN16 unsigned int select_y[4] = {0, 0xffffffff, 0, 0};
        ALIGN16 unsigned int select_z[4] = {0, 0, 0xffffffff, 0};
        return Matrix4f(Vector4f(vec_sel(zero, scaleVec._m128, select_x)),
                        Vector4f(vec_sel(zero, scaleVec._m128, select_y)),
                        Vector4f(vec_sel(zero, scaleVec._m128, select_z)),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline const Matrix4f translation(const Vector3f& translateVec) {
        return Matrix4f(Vector4f(_mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f)),
                        Vector4f(_mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)),
                        Vector4f(_mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f)),
                        Vector4f(translateVec, 1.0f));
    }

    inline void set_identity() {
        *this = Matrix4f(Vector4f(_mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f)),
                         Vector4f(_mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)),
                         Vector4f(_mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f)),
                         Vector4f(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    inline void prepend(const Matrix4f& myMatrix) {
        *this = myMatrix * (*this);
    }

    inline void append(const Matrix4f& myMatrix) {
        *this = (*this) * myMatrix;
    }

    inline void transpose() {
        __m128 tmp0, tmp1, tmp2, tmp3, res0, res1, res2, res3;
        tmp0 = vec_mergeh(m_Col0._m128, m_Col2._m128);
        tmp1 = vec_mergeh(m_Col1._m128, m_Col3._m128);
        tmp2 = vec_mergel(m_Col0._m128, m_Col2._m128);
        tmp3 = vec_mergel(m_Col1._m128, m_Col3._m128);
        res0 = vec_mergeh(tmp0, tmp1);
        res1 = vec_mergel(tmp0, tmp1);
        res2 = vec_mergeh(tmp2, tmp3);
        res3 = vec_mergel(tmp2, tmp3);
        *this = Matrix4f(Vector4f(res0), Vector4f(res1), Vector4f(res2),
                         Vector4f(res3));
    }

    inline bool has_inverse() const {
        return (fabs(determinant()) > FLOAT_EPSILON);
    }

    inline Matrix4f inverse() const {
        __m128 Va, Vb, Vc;
        __m128 r1, r2, r3, tt, tt2;
        __m128 sum, Det, RDet;
        __m128 trns0, trns1, trns2, trns3;

        __m128 _L1 = m_Col0._m128;
        __m128 _L2 = m_Col1._m128;
        __m128 _L3 = m_Col2._m128;
        __m128 _L4 = m_Col3._m128;
        // Calculating the minterms for the first line.

        // _mm_ror_ps is just a macro using _mm_shuffle_ps().
        tt = _L4;
        tt2 = _mm_ror_ps(_L3, 1);
        Vc = _mm_mul_ps(tt2, _mm_ror_ps(tt, 0)); // V3'dot V4
        Va = _mm_mul_ps(tt2, _mm_ror_ps(tt, 2)); // V3'dot V4"
        Vb = _mm_mul_ps(tt2, _mm_ror_ps(tt, 3)); // V3' dot V4^

        r1 = _mm_sub_ps(_mm_ror_ps(Va, 1),
                        _mm_ror_ps(Vc, 2)); // V3" dot V4^ - V3^ dot V4"
        r2 = _mm_sub_ps(_mm_ror_ps(Vb, 2),
                        _mm_ror_ps(Vb, 0)); // V3^ dot V4' - V3' dot V4^
        r3 = _mm_sub_ps(_mm_ror_ps(Va, 0),
                        _mm_ror_ps(Vc, 1)); // V3' dot V4" - V3" dot V4'

        tt = _L2;
        Va = _mm_ror_ps(tt, 1);
        sum = _mm_mul_ps(Va, r1);
        Vb = _mm_ror_ps(tt, 2);
        sum = _mm_add_ps(sum, _mm_mul_ps(Vb, r2));
        Vc = _mm_ror_ps(tt, 3);
        sum = _mm_add_ps(sum, _mm_mul_ps(Vc, r3));

        // Calculating the determinant.
        Det = _mm_mul_ps(sum, _L1);
        Det = _mm_add_ps(Det, _mm_movehl_ps(Det, Det));

        ALIGN16 const unsigned int _vmathPNPN[4] = {0x00000000, 0x80000000,
                                                    0x00000000, 0x80000000
                                                   };
        ALIGN16 const unsigned int _vmathNPNP[4] = {0x80000000, 0x00000000,
                                                    0x80000000, 0x00000000
                                                   };
        const __m128 Sign_PNPN = _mm_load_ps((float*)_vmathPNPN);
        const __m128 Sign_NPNP = _mm_load_ps((float*)_vmathNPNP);

        __m128 mtL1 = _mm_xor_ps(sum, Sign_PNPN);

        // Calculating the minterms of the second line (using previous results).
        tt = _mm_ror_ps(_L1, 1);
        sum = _mm_mul_ps(tt, r1);
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r2));
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r3));
        __m128 mtL2 = _mm_xor_ps(sum, Sign_NPNP);

        // Testing the determinant.
        Det = _mm_sub_ss(Det, _mm_shuffle_ps(Det, Det, 1));

        // Calculating the minterms of the third line.
        tt = _mm_ror_ps(_L1, 1);
        Va = _mm_mul_ps(tt, Vb);  // V1' dot V2"
        Vb = _mm_mul_ps(tt, Vc);  // V1' dot V2^
        Vc = _mm_mul_ps(tt, _L2); // V1' dot V2

        r1 = _mm_sub_ps(_mm_ror_ps(Va, 1),
                        _mm_ror_ps(Vc, 2)); // V1" dot V2^ - V1^ dot V2"
        r2 = _mm_sub_ps(_mm_ror_ps(Vb, 2),
                        _mm_ror_ps(Vb, 0)); // V1^ dot V2' - V1' dot V2^
        r3 = _mm_sub_ps(_mm_ror_ps(Va, 0),
                        _mm_ror_ps(Vc, 1)); // V1' dot V2" - V1" dot V2'

        tt = _mm_ror_ps(_L4, 1);
        sum = _mm_mul_ps(tt, r1);
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r2));
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r3));
        __m128 mtL3 = _mm_xor_ps(sum, Sign_PNPN);

        // Dividing is FASTER than rcp_nr! (Because rcp_nr causes many
        // register-memory RWs).
        ALIGN16 const float _vmathZERONE[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        RDet = _mm_div_ss(_mm_load_ss((float*)&_vmathZERONE),
                          Det); // TODO: just 1.0f?
        RDet = _mm_shuffle_ps(RDet, RDet, 0x00);

        // Devide the first 12 minterms with the determinant.
        mtL1 = _mm_mul_ps(mtL1, RDet);
        mtL2 = _mm_mul_ps(mtL2, RDet);
        mtL3 = _mm_mul_ps(mtL3, RDet);

        // Calculate the minterms of the forth line and devide by the determinant.
        tt = _mm_ror_ps(_L3, 1);
        sum = _mm_mul_ps(tt, r1);
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r2));
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r3));
        __m128 mtL4 = _mm_xor_ps(sum, Sign_NPNP);
        mtL4 = _mm_mul_ps(mtL4, RDet);

        // Now we just have to transpose the minterms matrix.
        trns0 = _mm_unpacklo_ps(mtL1, mtL2);
        trns1 = _mm_unpacklo_ps(mtL3, mtL4);
        trns2 = _mm_unpackhi_ps(mtL1, mtL2);
        trns3 = _mm_unpackhi_ps(mtL3, mtL4);
        _L1 = _mm_movelh_ps(trns0, trns1);
        _L2 = _mm_movehl_ps(trns1, trns0);
        _L3 = _mm_movelh_ps(trns2, trns3);
        _L4 = _mm_movehl_ps(trns3, trns2);

        return Matrix4f(Vector4f(_L1), Vector4f(_L2), Vector4f(_L3), Vector4f(_L4));
    }

    inline Vector3f transform_point(const Vector3f& mypoint) const {
        Vector4f spoint = Vector4f(mypoint);
        spoint._m[3] = 1.0f;
        Vector4f temp = *this * spoint;
        float dw = temp.get_elem(3);
        dw = fabs(dw) > FLOAT_EPSILON ? dw : FLOAT_EPSILON;
        dw = 1.0f / dw;
        temp = temp * dw;
        return Vector3f(temp._m128);
    }

    inline Vector3f transform_vector(const Vector3f& vec) const {
        __m128 res;
        __m128 xxxx, yyyy, zzzz;
        xxxx = vec_splat(vec._m128, 0);
        yyyy = vec_splat(vec._m128, 1);
        zzzz = vec_splat(vec._m128, 2);
        res = vec_mul(m_Col0._m128, xxxx);
        res = vec_madd(m_Col1._m128, yyyy, res);
        res = vec_madd(m_Col2._m128, zzzz, res);
        return Vector3f(res);
    }

    inline void extract_translate(Vector3f& myVector) const {
        myVector = Vector3f(m_Col3._m128);
    }

    inline float determinant() const {
        __m128 Va, Vb, Vc;
        __m128 r1, r2, r3, tt, tt2;
        __m128 sum, Det;

        __m128 _L1 = m_Col0._m128;
        __m128 _L2 = m_Col1._m128;
        __m128 _L3 = m_Col2._m128;
        __m128 _L4 = m_Col3._m128;
        // Calculating the minterms for the first line.

        // _mm_ror_ps is just a macro using _mm_shuffle_ps().
        tt = _L4;
        tt2 = _mm_ror_ps(_L3, 1);
        Vc = _mm_mul_ps(tt2, _mm_ror_ps(tt, 0)); // V3' dot V4
        Va = _mm_mul_ps(tt2, _mm_ror_ps(tt, 2)); // V3' dot V4"
        Vb = _mm_mul_ps(tt2, _mm_ror_ps(tt, 3)); // V3' dot V4^

        r1 = _mm_sub_ps(_mm_ror_ps(Va, 1),
                        _mm_ror_ps(Vc, 2)); // V3" dot V4^ - V3^ dot V4"
        r2 = _mm_sub_ps(_mm_ror_ps(Vb, 2),
                        _mm_ror_ps(Vb, 0)); // V3^ dot V4' - V3' dot V4^
        r3 = _mm_sub_ps(_mm_ror_ps(Va, 0),
                        _mm_ror_ps(Vc, 1)); // V3' dot V4" - V3" dot V4'

        tt = _L2;
        Va = _mm_ror_ps(tt, 1);
        sum = _mm_mul_ps(Va, r1);
        Vb = _mm_ror_ps(tt, 2);
        sum = _mm_add_ps(sum, _mm_mul_ps(Vb, r2));
        Vc = _mm_ror_ps(tt, 3);
        sum = _mm_add_ps(sum, _mm_mul_ps(Vc, r3));

        // Calculating the determinant.
        Det = _mm_mul_ps(sum, _L1);
        Det = _mm_add_ps(Det, _mm_movehl_ps(Det, Det));

        // Calculating the minterms of the second line (using previous results).
        tt = _mm_ror_ps(_L1, 1);
        sum = _mm_mul_ps(tt, r1);
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r2));
        tt = _mm_ror_ps(tt, 1);
        sum = _mm_add_ps(sum, _mm_mul_ps(tt, r3));

        // Testing the determinant.
        Det = _mm_sub_ss(Det, _mm_shuffle_ps(Det, Det, 1));
        return _vmathVfGetElement(Det, 0);
    }

    inline Matrix4f get_transpose() const {
        Matrix4f tmp = *this;
        tmp.transpose();
        return tmp;
    }

    inline bool operator==(const Matrix4f& myMatrix) const {
        return (m_Col0 == myMatrix.get_col0() && m_Col1 == myMatrix.get_col1() &&
                m_Col2 == myMatrix.get_col2() && m_Col3 == myMatrix.get_col3());
    }

    inline bool operator!=(const Matrix4f& myMatrix) const {
        return (m_Col0 != myMatrix.get_col0() || m_Col1 != myMatrix.get_col1() ||
                m_Col2 != myMatrix.get_col2() || m_Col3 != myMatrix.get_col3());
    }

    inline bool is_affine() const {
        Vector4f tmp = this->get_row(3);
        return (tmp == Vector4f(FLOAT_EPSILON, FLOAT_EPSILON, FLOAT_EPSILON,
                                FLOAT_EPSILON + 1.0f));
    }

    inline Matrix4f make_scale(const Vector3f& myScale) {
        set_identity();
        m_Col0 *= myScale.get_x();
        m_Col1 *= myScale.get_y();
        m_Col2 *= myScale.get_z();
        return *this;
    }

    inline Matrix4f make_translate(const Vector3f& myTranslate) {
        set_identity();
        m_Col3 += Vector4f(myTranslate, 0.0f);
        return *this;
    }
};

//////////////////////////////////////////////////////////////////////////
////Non-member function
//////////////////////////////////////////////////////////////////////////

// append (post-multiply) a scale transformation to a 4x4 matrix
// Faster than creating and multiplying a scale transformation matrix.
Arithmetic_Export inline const Matrix4f append_scale(const Matrix4f& mat,
        const Vector3f& scaleVec);

// prepend (pre-multiply) a scale transformation to a 4x4 matrix
// Faster than creating and multiplying a scale transformation matrix.
Arithmetic_Export inline const Matrix4f prepend_scale(const Vector3f& scaleVec,
        const Matrix4f& mat);

// Multiply two 4x4 matrices per element
Arithmetic_Export inline const Matrix4f mul_per_elem(const Matrix4f& mat0,
        const Matrix4f& mat1);

// Compute the absolute value of a 4x4 matrix per element
Arithmetic_Export inline const Matrix4f abs_per_elem(const Matrix4f& mat);

// transpose of a 4x4 matrix
Arithmetic_Export inline const Matrix4f transpose(const Matrix4f& mat);

// Compute the inverse of a 4x4 matrix
// Result is unpredictable when the determinant of mat is equal to or near 0.
Arithmetic_Export inline const Matrix4f inverse(const Matrix4f& mat);

// Compute the inverse of a 4x4 matrix, which is expected to be an affine matrix
// This can be used to achieve better performance than a general inverse when
// the specified 4x4 matrix meets the given restrictions.  The result is
// unpredictable when the determinant of mat is equal to or near 0.
Arithmetic_Export inline const Matrix4f affine_inverse(const Matrix4f& mat);

// Compute the inverse of a 4x4 matrix, which is expected to be an affine matrix
// with an orthogonal upper-left 3x3 submatrix
// This can be used to achieve better performance than a general inverse when
// the specified 4x4 matrix meets the given restrictions.
Arithmetic_Export inline const Matrix4f ortho_inverse(const Matrix4f& mat);

// determinant of a 4x4 matrix
Arithmetic_Export inline const float determinant(const Matrix4f& mat);


MED_IMG_END_NAMESPACE

#endif