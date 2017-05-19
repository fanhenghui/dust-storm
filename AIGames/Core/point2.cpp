#include "point2.h"



const Point2 Point2::S_ZERO_POINT = Point2(0, 0);

Point2 operator*(double scale, const Point2& p)
{
    return Point2(p.x * scale, p.y * scale);
}

Vector2 operator -(const Point2& pt1, const Point2& pt2)
{
    return Vector2(pt1.x - pt2.x, pt1.y - pt2.y);
}

Point2 operator +(const Point2& pt, const Vector2& v)
{
    return Point2(pt.x + v.x, pt.y + v.y);
}

Point2 operator -(const Point2& pt, const Vector2& v)
{
    return Point2(pt.x - v.x, pt.y - v.y);
}


