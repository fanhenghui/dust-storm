#include "mi_vector2.h"

MED_IMAGING_BEGIN_NAMESPACE

const Vector2 Vector2::kZeroVector = Vector2(0, 0);

Vector2 operator *(double scale, const Vector2& v)
{
    return Vector2(scale * v.x, scale * v.y);
}

double AngleBetween(const Vector2& v1,const Vector2& v2)
{
    double len = v1.Magnitude() * v2.Magnitude();

    len = len > DOUBLE_EPSILON ? len : DOUBLE_EPSILON;

    double dot = v1.DotProduct(v2) / len;
    dot = std::min(dot, 1.0);
    dot = std::max(dot, -1.0);
    return std::acos(dot);
}

double DotProduct(const Vector2& v1 , const Vector2& v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

bool Parallel(const Vector2& v1, const Vector2& v2)
{
    return std::fabs(v1.DotProduct(v2) - v1.Magnitude()*v2.Magnitude()) < DOUBLE_EPSILON;
}

bool Orthogonal(const Vector2& v1, const Vector2& v2)
{
    return std::fabs(v1.DotProduct(v2)) < DOUBLE_EPSILON;
}

MED_IMAGING_END_NAMESPACE