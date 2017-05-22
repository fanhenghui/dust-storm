#ifndef MED_IMAGING_LINE_SEGMENT_H
#define MED_IMAGING_LINE_SEGMENT_H

#include "Core/shape_interface.h"
#include "Core/point2.h"
#include "Core/point3.h"

class  LineSegment2D : public IShape
{
public:
    Point2 _pt[2];

public:
    LineSegment2D();
    virtual ~LineSegment2D();
};

class  LineSegment3D : public IShape
{
public:
    Point3 _pt[2];

public:
    LineSegment3D();
    LineSegment3D(const Point3& pt0 , const Point3& pt1);
    virtual ~LineSegment3D();
};



#endif