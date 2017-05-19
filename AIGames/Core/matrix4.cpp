#include "matrix4.h"



    const Matrix4 Matrix4::S_ZERO_MATRIX = Matrix4(
    0,0,0,0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0);

const Matrix4 Matrix4::S_IDENTITY_MATRIX = Matrix4(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1);

Matrix4 make_scale(const Vector3 & v)
{
    return Matrix4(
        v.x, 0.0, 0.0, 0.0,
        0.0, v.y, 0.0, 0.0,
        0.0, 0.0, v.z, 0.0,
        0.0, 0.0, 0.0, 1.0);
}

Matrix4 make_translate(const Vector3 &v)
{
    return Matrix4(1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        v.x, v.y, v.z, 1.0);
}

