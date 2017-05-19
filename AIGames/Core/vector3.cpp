#include "vector3.h"



const Vector3 Vector3::S_ZERO_VECTOR = Vector3(0, 0,0);

Vector3 operator *(double scale, const Vector3& v)
{
    return Vector3(scale * v.x, scale * v.y,scale * v.z);
}

double angle_between(const Vector3& v1,const Vector3& v2)
{
    double len = v1.magnitude() * v2.magnitude();

    len = len > DOUBLE_EPSILON ? len : DOUBLE_EPSILON;

    double dot = v1.dot_product(v2) / len;
    dot = std::min(dot, 1.0);
    dot = std::max(dot, -1.0);
    return std::acos(dot);
}

Vector3 cross(const Vector3& v1, const Vector3& v2)
{
    return Vector3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

double dot_product(const Vector3& v1,const Vector3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

bool parallel(const Vector3& v1, const Vector3& v2)
{
    return v1.cross_product(v2) == Vector3(0, 0, 0);
}

bool orthogonal(const Vector3& v1, const Vector3& v2)
{
    return std::fabs(v1.dot_product(v2)) < DOUBLE_EPSILON;
}


