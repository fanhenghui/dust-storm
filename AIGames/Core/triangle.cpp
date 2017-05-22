#include "triangle.h"



Triangle::Triangle()
{

}

Triangle::Triangle(Point3 pt0 , Point3 pt1 ,Point3 pt2)
{
    _pt[0] = pt0;
    _pt[1] = pt1;
    _pt[2] = pt2;
}

void Triangle::to_lines(std::vector<LineSegment3D> &lines) const
{
    lines.clear();
    lines.resize(3);
    lines[0] = LineSegment3D(_pt[0] , _pt[1]);
    lines[1] = LineSegment3D(_pt[1] , _pt[2]);
    lines[2] = LineSegment3D(_pt[2] , _pt[0]);
}

Vector3 Triangle::get_normal() const
{
    Vector3 v01 = _pt[1] - _pt[0];
    Vector3 v02  = _pt[2] - _pt[0];
    Vector3 norm = v01.cross_product(v02);
    return norm.get_normalize();
}

bool Triangle::in_triangle(const Point3 &pt0)
{
    //P点在ABC内的方法可以用PAB PBC PCA呈现顺时针排列或者逆时针排列 ，计算差乘结果判断方向一致性
    const Vector3 a = _pt[0] - pt0;
    const Vector3 b = _pt[1] - pt0;
    const Vector3 c = _pt[2] - pt0;

    const Vector3 u = b.cross_product(c);
    const Vector3 v = c.cross_product(a);

    if (u.dot_product(v) < 0.0)
    {
        return false;
    }
    else
    {
        Vector3 w = a.cross_product(b);
        if (u.dot_product(w) < 0.0)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

