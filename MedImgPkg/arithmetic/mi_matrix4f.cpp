#include "mi_matrix4f.h"

MED_IMG_BEGIN_NAMESPACE

const Matrix4f append_scale(const Matrix4f& mat, const Vector3f& scaleVec) {
    return Matrix4f((mat.get_col0() * scaleVec.get_x()),
                    (mat.get_col1() * scaleVec.get_y()),
                    (mat.get_col2() * scaleVec.get_z()), mat.get_col3());
}

const Matrix4f prepend_scale(const Vector3f& scaleVec, const Matrix4f& mat) {
    Vector4f scale4 = Vector4f(scaleVec, 1.0f);
    return Matrix4f(mul_per_elem(mat.get_col0(), scale4),
                    mul_per_elem(mat.get_col1(), scale4),
                    mul_per_elem(mat.get_col2(), scale4),
                    mul_per_elem(mat.get_col3(), scale4));
}

const Matrix4f mul_per_elem(const Matrix4f& mat0, const Matrix4f& mat1) {
    return Matrix4f(mul_per_elem(mat0.get_col0(), mat1.get_col0()),
                    mul_per_elem(mat0.get_col1(), mat1.get_col1()),
                    mul_per_elem(mat0.get_col2(), mat1.get_col2()),
                    mul_per_elem(mat0.get_col3(), mat1.get_col3()));
}

const Matrix4f abs_per_elem(const Matrix4f& mat) {
    return Matrix4f(abs_per_elem(mat.get_col0()), abs_per_elem(mat.get_col1()),
                    abs_per_elem(mat.get_col2()), abs_per_elem(mat.get_col3()));
}

const Matrix4f transpose(const Matrix4f& mat) {
    __m128 tmp0 = vec_mergeh(mat.get_col0()._m128, mat.get_col2()._m128);
    __m128 tmp1 = vec_mergeh(mat.get_col1()._m128, mat.get_col3()._m128);
    __m128 tmp2 = vec_mergel(mat.get_col0()._m128, mat.get_col2()._m128);
    __m128 tmp3 = vec_mergel(mat.get_col1()._m128, mat.get_col3()._m128);
    __m128 res0 = vec_mergeh(tmp0, tmp1);
    __m128 res1 = vec_mergel(tmp0, tmp1);
    __m128 res2 = vec_mergeh(tmp2, tmp3);
    __m128 res3 = vec_mergel(tmp2, tmp3);
    return Matrix4f(Vector4f(res0), Vector4f(res1), Vector4f(res2),
                    Vector4f(res3));
}

const Matrix4f inverse(const Matrix4f& mat) {
    __m128 _L1 = mat.get_col0()._m128;
    __m128 _L2 = mat.get_col1()._m128;
    __m128 _L3 = mat.get_col2()._m128;
    __m128 _L4 = mat.get_col3()._m128;
    // Calculating the minterms for the first line.

    // _mm_ror_ps is just a macro using _mm_shuffle_ps().
    __m128 tt = _L4;
    __m128 tt2 = _mm_ror_ps(_L3, 1);
    __m128 Vc = _mm_mul_ps(tt2, _mm_ror_ps(tt, 0)); // V3'dot V4
    __m128 Va = _mm_mul_ps(tt2, _mm_ror_ps(tt, 2)); // V3'dot V4"
    __m128 Vb = _mm_mul_ps(tt2, _mm_ror_ps(tt, 3)); // V3' dot V4^

    __m128 r1 = _mm_sub_ps(_mm_ror_ps(Va, 1),
                           _mm_ror_ps(Vc, 2)); // V3" dot V4^ - V3^ dot V4"
    __m128 r2 = _mm_sub_ps(_mm_ror_ps(Vb, 2),
                           _mm_ror_ps(Vb, 0)); // V3^ dot V4' - V3' dot V4^
    __m128 r3 = _mm_sub_ps(_mm_ror_ps(Va, 0),
                           _mm_ror_ps(Vc, 1)); // V3' dot V4" - V3" dot V4'

    tt = _L2;
    Va = _mm_ror_ps(tt, 1);
    __m128 sum = _mm_mul_ps(Va, r1);
    Vb = _mm_ror_ps(tt, 2);
    sum = _mm_add_ps(sum, _mm_mul_ps(Vb, r2));
    Vc = _mm_ror_ps(tt, 3);
    sum = _mm_add_ps(sum, _mm_mul_ps(Vc, r3));

    // Calculating the determinant.
    __m128 Det = _mm_mul_ps(sum, _L1);
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

    // Dividing is FASTER than rcp_nr! (Because rcp_nr causes many register-memory
    // RWs).
    ALIGN16 const float _vmathZERONE[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    __m128 RDet =
        _mm_div_ss(_mm_load_ss((float*)&_vmathZERONE), Det);  // TODO: just 1.0f?
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
    __m128 trns0 = _mm_unpacklo_ps(mtL1, mtL2);
    __m128 trns1 = _mm_unpacklo_ps(mtL3, mtL4);
    __m128 trns2 = _mm_unpackhi_ps(mtL1, mtL2);
    __m128 trns3 = _mm_unpackhi_ps(mtL3, mtL4);
    _L1 = _mm_movelh_ps(trns0, trns1);
    _L2 = _mm_movehl_ps(trns1, trns0);
    _L3 = _mm_movelh_ps(trns2, trns3);
    _L4 = _mm_movehl_ps(trns3, trns2);

    return Matrix4f(Vector4f(_L1), Vector4f(_L2), Vector4f(_L3), Vector4f(_L4));
}

const Matrix4f affine_inverse(const Matrix4f& mat) {
    Vector3f col0 = mat.get_col0().get_xyz();
    Vector3f col1 = mat.get_col1().get_xyz();
    Vector3f col2 = mat.get_col2().get_xyz();
    Vector3f col3 = mat.get_col3().get_xyz();

    Vector3f tmp0 = cross(col1, col2);
    Vector3f tmp1 = cross(col2, col0);
    Vector3f tmp2 = cross(col0, col1);
    float detinv = (1.0f / dot_product(col2, tmp2));

    Vector3f inv0 = Vector3f((tmp0.get_x() * detinv), (tmp1.get_x() * detinv),
                             (tmp2.get_x() * detinv));
    Vector3f inv1 = Vector3f((tmp0.get_y() * detinv), (tmp1.get_y() * detinv),
                             (tmp2.get_y() * detinv));
    Vector3f inv2 = Vector3f((tmp0.get_z() * detinv), (tmp1.get_z() * detinv),
                             (tmp2.get_z() * detinv));

    Vector3f temp = Vector3f(-((inv0 * col3.get_x()) +
                               ((inv1 * col3.get_y()) + (inv2 * col3.get_z()))));

    return Matrix4f(Vector4f(inv0, 0.0f), Vector4f(inv1, 0.0f),
                    Vector4f(inv2, 0.0f), Vector4f(temp, 1.0));
}

const Matrix4f ortho_inverse(const Matrix4f& mat) {
    Vector3f inv0 = mat.get_row(0).get_xyz();
    Vector3f inv1 = mat.get_row(1).get_xyz();
    Vector3f inv2 = mat.get_row(2).get_xyz();
    Vector3f temp = Vector3f(
                        -((inv0 * mat.get_col3().get_x()) +
                          ((inv1 * mat.get_col3().get_y()) + (inv2 * mat.get_col3().get_z()))));
    return Matrix4f(Vector4f(inv0, 0.0f), Vector4f(inv1, 0.0f),
                    Vector4f(inv2, 0.0f), Vector4f(temp, 1.0));
}

const float determinant(const Matrix4f& mat) {
    __m128 _L1 = mat.get_col0()._m128;
    __m128 _L2 = mat.get_col1()._m128;
    __m128 _L3 = mat.get_col2()._m128;
    __m128 _L4 = mat.get_col3()._m128;
    // Calculating the minterms for the first line.

    // _mm_ror_ps is just a macro using _mm_shuffle_ps().
    __m128 tt = _L4;
    __m128 tt2 = _mm_ror_ps(_L3, 1);
    __m128 Vc = _mm_mul_ps(tt2, _mm_ror_ps(tt, 0)); // V3' dot V4
    __m128 Va = _mm_mul_ps(tt2, _mm_ror_ps(tt, 2)); // V3' dot V4"
    __m128 Vb = _mm_mul_ps(tt2, _mm_ror_ps(tt, 3)); // V3' dot V4^

    __m128 r1 = _mm_sub_ps(_mm_ror_ps(Va, 1),
                           _mm_ror_ps(Vc, 2)); // V3" dot V4^ - V3^ dot V4"
    __m128 r2 = _mm_sub_ps(_mm_ror_ps(Vb, 2),
                           _mm_ror_ps(Vb, 0)); // V3^ dot V4' - V3' dot V4^
    __m128 r3 = _mm_sub_ps(_mm_ror_ps(Va, 0),
                           _mm_ror_ps(Vc, 1)); // V3' dot V4" - V3" dot V4'

    tt = _L2;
    Va = _mm_ror_ps(tt, 1);
    __m128 sum = _mm_mul_ps(Va, r1);
    Vb = _mm_ror_ps(tt, 2);
    sum = _mm_add_ps(sum, _mm_mul_ps(Vb, r2));
    Vc = _mm_ror_ps(tt, 3);
    sum = _mm_add_ps(sum, _mm_mul_ps(Vc, r3));

    // Calculating the determinant.
    __m128 Det = _mm_mul_ps(sum, _L1);
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

MED_IMG_END_NAMESPACE