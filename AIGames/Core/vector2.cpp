#include "vector2.h"



const Vector2 Vector2::S_ZERO_VECTOR = Vector2(0, 0);

Vector2 operator *(double scale, const Vector2& v)
{
    return Vector2(scale * v.x, scale * v.y);
}

double angle_between(const Vector2& v1,const Vector2& v2)
{
    double len = v1.magnitude() * v2.magnitude();

    len = len > DOUBLE_EPSILON ? len : DOUBLE_EPSILON;

    double dot = v1.dot_product(v2) / len;
    dot = std::min(dot, 1.0);
    dot = std::max(dot, -1.0);
    return std::acos(dot);
}

double dot_product(const Vector2& v1 , const Vector2& v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

bool parallel(const Vector2& v1, const Vector2& v2)
{
    return std::fabs(v1.dot_product(v2) - v1.magnitude()*v2.magnitude()) < DOUBLE_EPSILON;
}

bool orthogonal(const Vector2& v1, const Vector2& v2)
{
    return std::fabs(v1.dot_product(v2)) < DOUBLE_EPSILON;
}

