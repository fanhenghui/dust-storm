#include "point3.h"



const Point3 Point3::S_ZERO_POINT = Point3(0, 0, 0);

Point3 operator*(double scale, const Point3& p)
{
    return Point3(p.x * scale, p.y * scale, p.z * scale);
}

Vector3 operator -(const Point3& pt1, const Point3& pt2)
{
    return Vector3(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
}

Point3 operator +(const Point3& pt, const Vector3& v)
{
    return Point3(pt.x + v.x, pt.y + v.y, pt.z + v.z);
}

Point3 operator -(const Point3& pt, const Vector3& v)
{
    return Point3(pt.x - v.x, pt.y - v.y, pt.z - v.z);
}



