#include "mi_line_segment.h"

MED_IMAGING_BEGIN_NAMESPACE

LineSegment2D::LineSegment2D()
{

}

LineSegment2D::~LineSegment2D()
{

}


LineSegment3D::LineSegment3D()
{

}

LineSegment3D::LineSegment3D(const Point3& pt0 , const Point3& pt1)
{
    m_pt[0] = pt0;
    m_pt[1] = pt1;
}

LineSegment3D::~LineSegment3D()
{

}

MED_IMAGING_END_NAMESPACE