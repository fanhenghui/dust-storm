#ifndef ARTHMETIC_MATRIX_4D_H_
#define ARTHMETIC_MATRIX_4D_H_

#include "Core/vector3.h"
#include "Core/point3.h"

class  Matrix4
{
public:
    union
    {
        double m[4][4];
        double _m[16];
    };

    static const Matrix4 S_ZERO_MATRIX;
    static const Matrix4 S_IDENTITY_MATRIX;

public:

    Matrix4::Matrix4()
    {
    }

    Matrix4::Matrix4(double m00, double m01, double m02, double m03,
        double m10, double m11, double m12, double m13,
        double m20, double m21, double m22, double m23,
        double m30, double m31, double m32, double m33)
    {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;
        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    Matrix4::Matrix4(const Matrix4& mat)
    {
        memcpy(this->_m, mat._m, sizeof(mat._m));
    }

    inline double* Matrix4::operator[] (size_t iRow)
    {
        if (iRow > 4)
            return nullptr;
        else
            return m[iRow];
    }

    inline const double *Matrix4::operator [] (size_t iRow) const
    {
        if (iRow > 4)
            return nullptr;
        else
            return m[iRow];
    }

    inline Matrix4& Matrix4::operator = (const Matrix4& mat)
    {
        for (int i = 0 ; i<16 ; ++i)
        {
            _m[i] = mat._m[i];
        }
        return *this;
    }

    inline Matrix4 Matrix4::operator * (const Matrix4 & m2) const
    {
        return Matrix4(
            _m[0] * m2._m[0] + _m[4] * m2._m[1] + _m[8] * m2._m[2] + _m[12] * m2._m[3],
            _m[1] * m2._m[0] + _m[5] * m2._m[1] + _m[9] * m2._m[2] + _m[13] * m2._m[3],
            _m[2] * m2._m[0] + _m[6] * m2._m[1] + _m[10] * m2._m[2] + _m[14] * m2._m[3],
            _m[3] * m2._m[0] + _m[7] * m2._m[1] + _m[11] * m2._m[2] + _m[15] * m2._m[3],
            _m[0] * m2._m[4] + _m[4] * m2._m[5] + _m[8] * m2._m[6] + _m[12] * m2._m[7],
            _m[1] * m2._m[4] + _m[5] * m2._m[5] + _m[9] * m2._m[6] + _m[13] * m2._m[7],
            _m[2] * m2._m[4] + _m[6] * m2._m[5] + _m[10] * m2._m[6] + _m[14] * m2._m[7],
            _m[3] * m2._m[4] + _m[7] * m2._m[5] + _m[11] * m2._m[6] + _m[15] * m2._m[7],
            _m[0] * m2._m[8] + _m[4] * m2._m[9] + _m[8] * m2._m[10] + _m[12] * m2._m[11],
            _m[1] * m2._m[8] + _m[5] * m2._m[9] + _m[9] * m2._m[10] + _m[13] * m2._m[11],
            _m[2] * m2._m[8] + _m[6] * m2._m[9] + _m[10] * m2._m[10] + _m[14] * m2._m[11],
            _m[3] * m2._m[8] + _m[7] * m2._m[9] + _m[11] * m2._m[10] + _m[15] * m2._m[11],
            _m[0] * m2._m[12] + _m[4] * m2._m[13] + _m[8] * m2._m[14] + _m[12] * m2._m[15],
            _m[1] * m2._m[12] + _m[5] * m2._m[13] + _m[9] * m2._m[14] + _m[13] * m2._m[15],
            _m[2] * m2._m[12] + _m[6] * m2._m[13] + _m[10] * m2._m[14] + _m[14] * m2._m[15],
            _m[3] * m2._m[12] + _m[7] * m2._m[13] + _m[11] * m2._m[14] + _m[15] * m2._m[15]
        );
    }

    inline Matrix4 Matrix4::operator *= (const Matrix4 & m2)
    {
        (*this) = (*this)*m2;
        return *this;
    }

    inline Vector3 Matrix4::operator * (const Vector3 &v) const
    {
        return  Vector3(_m[0] * v.x + _m[4] * v.y + _m[8] * v.z,
            _m[1] * v.x + _m[5] * v.y + _m[9] * v.z,
            _m[2] * v.x + _m[6] * v.y + _m[10] * v.z);
    }

    inline Point3 Matrix4::operator * (const Point3& pt) const
    {
        double dw = _m[3] * pt.x + _m[7] * pt.y + _m[11] * pt.z + _m[15];
        dw = fabs(dw) > DOUBLE_EPSILON ? dw : DOUBLE_EPSILON;
        dw = 1.0 / dw;
        return  Point3(
            (_m[0] * pt.x + _m[4] * pt.y + _m[8] * pt.z + _m[12]) * dw,
            (_m[1] * pt.x + _m[5] * pt.y + _m[9] * pt.z + _m[13]) * dw,
            (_m[2] * pt.x + _m[6] * pt.y + _m[10] * pt.z + _m[14]) * dw);
    }

    inline Matrix4 Matrix4::operator * (const double scale) const
    {
        return Matrix4(
            scale * _m[0], scale * _m[1], scale * _m[2], scale *_m[3],
            scale * _m[4], scale * _m[5], scale * _m[6], scale *_m[7],
            scale * _m[8], scale * _m[9], scale * _m[10], scale *_m[11],
            scale * _m[12], scale * _m[13], scale * _m[14], scale *_m[15]
        );
    }

    inline Matrix4 Matrix4::operator *= (const double scale)
    {
        for (int i = 0; i < 16; ++i)
        {
            _m[i] *= scale;
        }
        return *this;
    }

    inline bool Matrix4::operator == (const Matrix4& mat) const
    {
        return (std::fabs(_m[0] - mat._m[0]) < DOUBLE_EPSILON &&
            std::fabs(_m[1] - mat._m[1]) < DOUBLE_EPSILON &&
            std::fabs(_m[2] - mat._m[2]) < DOUBLE_EPSILON &&
            std::fabs(_m[3] - mat._m[3]) < DOUBLE_EPSILON &&
            std::fabs(_m[4] - mat._m[4]) < DOUBLE_EPSILON &&
            std::fabs(_m[5] - mat._m[5]) < DOUBLE_EPSILON &&
            std::fabs(_m[6] - mat._m[6]) < DOUBLE_EPSILON &&
            std::fabs(_m[7] - mat._m[7]) < DOUBLE_EPSILON &&
            std::fabs(_m[8] - mat._m[8]) < DOUBLE_EPSILON &&
            std::fabs(_m[9] - mat._m[9]) < DOUBLE_EPSILON &&
            std::fabs(_m[10] - mat._m[10]) < DOUBLE_EPSILON &&
            std::fabs(_m[11] - mat._m[11]) < DOUBLE_EPSILON &&
            std::fabs(_m[12] - mat._m[12]) < DOUBLE_EPSILON &&
            std::fabs(_m[13] - mat._m[13]) < DOUBLE_EPSILON &&
            std::fabs(_m[14] - mat._m[14]) < DOUBLE_EPSILON &&
            std::fabs(_m[15] - mat._m[15]) < DOUBLE_EPSILON);
    }

    inline bool Matrix4::operator != (const Matrix4& mat) const
    {
        return (std::fabs(_m[0] - mat._m[0]) > DOUBLE_EPSILON ||
            std::fabs(_m[1] - mat._m[1]) > DOUBLE_EPSILON ||
            std::fabs(_m[2] - mat._m[2]) > DOUBLE_EPSILON ||
            std::fabs(_m[3] - mat._m[3]) > DOUBLE_EPSILON ||
            std::fabs(_m[4] - mat._m[4]) > DOUBLE_EPSILON ||
            std::fabs(_m[5] - mat._m[5]) > DOUBLE_EPSILON ||
            std::fabs(_m[6] - mat._m[6]) > DOUBLE_EPSILON ||
            std::fabs(_m[7] - mat._m[7]) > DOUBLE_EPSILON ||
            std::fabs(_m[8] - mat._m[8]) > DOUBLE_EPSILON ||
            std::fabs(_m[9] - mat._m[9]) > DOUBLE_EPSILON ||
            std::fabs(_m[10] - mat._m[10]) > DOUBLE_EPSILON ||
            std::fabs(_m[11] - mat._m[11]) > DOUBLE_EPSILON ||
            std::fabs(_m[12] - mat._m[12]) > DOUBLE_EPSILON ||
            std::fabs(_m[13] - mat._m[13]) > DOUBLE_EPSILON ||
            std::fabs(_m[14] - mat._m[14]) > DOUBLE_EPSILON ||
            std::fabs(_m[15] - mat._m[15]) > DOUBLE_EPSILON);
    }

    void Matrix4::set_idintity()
    {
        m[0][0] = 1.0; m[0][1] = 0.0; m[0][2] = 0.0; m[0][3] = 0.0;
        m[1][0] = 0.0; m[1][1] = 1.0; m[1][2] = 0.0; m[1][3] = 0.0;
        m[2][0] = 0.0; m[2][1] = 0.0; m[2][2] = 1.0; m[2][3] = 0.0;
        m[3][0] = 0.0; m[3][1] = 0.0; m[3][2] = 0.0; m[3][3] = 1.0;
    }

    inline void Matrix4::prepend(const Matrix4 &mat)
    {
        *this = mat * (*this);
    }

    inline void Matrix4::append(const Matrix4 &mat)
    {
        *this = (*this) * mat;
    }

    inline void Matrix4::transpose()
    {
        std::swap(_m[1], _m[4]);
        std::swap(_m[2], _m[8]);
        std::swap(_m[3], _m[12]);
        std::swap(_m[6], _m[9]);
        std::swap(_m[7], _m[13]);
        std::swap(_m[11], _m[14]);
    }

    inline Matrix4 Matrix4::get_transpose() const
    {
        return Matrix4(_m[0], _m[4], _m[8], _m[12],
            _m[1], _m[5], _m[9], _m[13],
            _m[2], _m[6], _m[10], _m[14],
            _m[3], _m[7], _m[11], _m[15]);
    }

    inline bool Matrix4::has_inverse() const
    {
        return (std::fabs(determinant()) > DOUBLE_EPSILON);

    }

    inline double Matrix4::determinant() const
    {
        double m00 = m[0][0], m01 = m[0][1], m02 = m[0][2], m03 = m[0][3];
        double m10 = m[1][0], m11 = m[1][1], m12 = m[1][2], m13 = m[1][3];
        double m20 = m[2][0], m21 = m[2][1], m22 = m[2][2], m23 = m[2][3];
        double m30 = m[3][0], m31 = m[3][1], m32 = m[3][2], m33 = m[3][3];

        double v0 = m20 * m31 - m21 * m30;
        double v1 = m20 * m32 - m22 * m30;
        double v2 = m20 * m33 - m23 * m30;
        double v3 = m21 * m32 - m22 * m31;
        double v4 = m21 * m33 - m23 * m31;
        double v5 = m22 * m33 - m23 * m32;

        double t00 = +(v5 * m11 - v4 * m12 + v3 * m13);
        double t10 = -(v5 * m10 - v2 * m12 + v1 * m13);
        double t20 = +(v4 * m10 - v2 * m11 + v0 * m13);
        double t30 = -(v3 * m10 - v1 * m11 + v0 * m12);

        return  (t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03);
    }

    inline Matrix4 Matrix4::get_inverse() const
    {
        double m00 = m[0][0], m01 = m[0][1], m02 = m[0][2], m03 = m[0][3];
        double m10 = m[1][0], m11 = m[1][1], m12 = m[1][2], m13 = m[1][3];
        double m20 = m[2][0], m21 = m[2][1], m22 = m[2][2], m23 = m[2][3];
        double m30 = m[3][0], m31 = m[3][1], m32 = m[3][2], m33 = m[3][3];

        double v0 = m20 * m31 - m21 * m30;
        double v1 = m20 * m32 - m22 * m30;
        double v2 = m20 * m33 - m23 * m30;
        double v3 = m21 * m32 - m22 * m31;
        double v4 = m21 * m33 - m23 * m31;
        double v5 = m22 * m33 - m23 * m32;

        double t00 = +(v5 * m11 - v4 * m12 + v3 * m13);
        double t10 = -(v5 * m10 - v2 * m12 + v1 * m13);
        double t20 = +(v4 * m10 - v2 * m11 + v0 * m13);
        double t30 = -(v3 * m10 - v1 * m11 + v0 * m12);

        double det = t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03;
        if (std::fabs(det) < DOUBLE_EPSILON)
        {
            return *this;
        }
        double invDet = 1 / det;

        double d00 = t00 * invDet;
        double d10 = t10 * invDet;
        double d20 = t20 * invDet;
        double d30 = t30 * invDet;

        double d01 = -(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        double d11 = +(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        double d21 = -(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        double d31 = +(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        v0 = m10 * m31 - m11 * m30;
        v1 = m10 * m32 - m12 * m30;
        v2 = m10 * m33 - m13 * m30;
        v3 = m11 * m32 - m12 * m31;
        v4 = m11 * m33 - m13 * m31;
        v5 = m12 * m33 - m13 * m32;

        double d02 = +(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        double d12 = -(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        double d22 = +(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        double d32 = -(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        v0 = m21 * m10 - m20 * m11;
        v1 = m22 * m10 - m20 * m12;
        v2 = m23 * m10 - m20 * m13;
        v3 = m22 * m11 - m21 * m12;
        v4 = m23 * m11 - m21 * m13;
        v5 = m23 * m12 - m22 * m13;

        double d03 = -(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        double d13 = +(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        double d23 = -(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        double d33 = +(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        return Matrix4(
            d00, d01, d02, d03,
            d10, d11, d12, d13,
            d20, d21, d22, d23,
            d30, d31, d32, d33);
    }

    inline Matrix4 Matrix4::inverse()
    {
        *this = get_inverse();
        return *this;
    }

    inline Point3 Matrix4::transform(const Point3 &pt) const
    {
        return (*this) * pt;
    }

    inline Vector3 Matrix4::transform(const Vector3 & v) const
    {
        return (*this) * v;
    }

    inline void to_float16(float (&fMat)[16]) const
    {
        for (int i = 0; i<16 ; ++i)
        {
            fMat[i] = (float)_m[i];
        }
    }
};

Matrix4  make_scale(const Vector3 & v);

Matrix4  make_translate(const Vector3 &v);



#endif