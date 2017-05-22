#include "line_segment.h"



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
    _pt[0] = pt0;
    _pt[1] = pt1;
}

LineSegment3D::~LineSegment3D()
{

}

