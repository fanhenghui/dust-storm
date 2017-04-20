#include "mi_vector3.h"

MED_IMAGING_BEGIN_NAMESPACE

const Vector3 Vector3::kZeroVector = Vector3(0, 0,0);

Vector3 operator *(double scale, const Vector3& v)
{
    return Vector3(scale * v.x, scale * v.y,scale * v.z);
}

double AngleBetween(const Vector3& v1,const Vector3& v2)
{
    double len = v1.Magnitude() * v2.Magnitude();

    len = len > DOUBLE_EPSILON ? len : DOUBLE_EPSILON;

    double dot = v1.DotProduct(v2) / len;
    dot = std::min(dot, 1.0);
    dot = std::max(dot, -1.0);
    return std::acos(dot);
}

Vector3 CrossProduct(const Vector3& v1, const Vector3& v2)
{
    return Vector3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

double DotProduct(const Vector3& v1,const Vector3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

bool Parallel(const Vector3& v1, const Vector3& v2)
{
    return v1.CrossProduct(v2) == Vector3(0, 0, 0);
}

bool Orthogonal(const Vector3& v1, const Vector3& v2)
{
    return std::fabs(v1.DotProduct(v2)) < DOUBLE_EPSILON;
}

MED_IMAGING_END_NAMESPACE