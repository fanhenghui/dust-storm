#include "mi_triangle.h"

MED_IMAGING_BEGIN_NAMESPACE

Triangle::Triangle()
{

}

Triangle::Triangle(Point3 pt0 , Point3 pt1 ,Point3 pt2)
{
    m_pt[0] = pt0;
    m_pt[1] = pt1;
    m_pt[2] = pt2;
}

void Triangle::GenerateLines(std::vector<LineSegment3D> &vLines) const
{
    vLines.clear();
    vLines.resize(3);
    vLines[0] = LineSegment3D(m_pt[0] , m_pt[1]);
    vLines[1] = LineSegment3D(m_pt[1] , m_pt[2]);
    vLines[2] = LineSegment3D(m_pt[2] , m_pt[0]);
}

Vector3 Triangle::GetNormal() const
{
    Vector3 v01 = m_pt[1] - m_pt[0];
    Vector3 v02  = m_pt[2] - m_pt[0];
    Vector3 vNorm = v01.CrossProduct(v02);
    return vNorm.GetNormalize();
}

bool Triangle::InTriangle(const Point3 &pt0)
{
    //P点在ABC内的方法可以用PAB PBC PCA呈现顺时针排列或者逆时针排列 ，计算差乘结果判断方向一致性
    const Vector3 a = m_pt[0] - pt0;
    const Vector3 b = m_pt[1] - pt0;
    const Vector3 c = m_pt[2] - pt0;

    const Vector3 u = b.CrossProduct(c);
    const Vector3 v = c.CrossProduct(a);

    if (u.DotProduct(v) < 0.0)
    {
        return false;
    }
    else
    {
        Vector3 w = a.CrossProduct(b);
        if (u.DotProduct(w) < 0.0)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

MED_IMAGING_END_NAMESPACE