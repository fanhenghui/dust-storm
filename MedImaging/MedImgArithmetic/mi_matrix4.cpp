#include "mi_matrix4.h"

MED_IMAGING_BEGIN_NAMESPACE

const Matrix4 Matrix4::kZeroMatrix = Matrix4(
	0,0,0,0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0);

const Matrix4 Matrix4::kIdentityMatrix = Matrix4(
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1);

Matrix4 MakeScale(const Vector3 & v)
{
	return Matrix4(
		v.x, 0.0, 0.0, 0.0,
		0.0, v.y, 0.0, 0.0,
		0.0, 0.0, v.z, 0.0,
		0.0, 0.0, 0.0, 1.0);
}

Matrix4 MakeTranslate(const Vector3 &v)
{
	return Matrix4(1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		v.x, v.y, v.z, 1.0);
}

MED_IMAGING_END_NAMESPACE