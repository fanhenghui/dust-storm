#ifndef MED_IMG_LINE_SEGMENT_H
#define MED_IMG_LINE_SEGMENT_H

#include "arithmetic/mi_shape_interface.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE 

class Arithmetic_Export LineSegment2D : public IShape
{
public:
    Point2 _pt[2];

public:
    LineSegment2D();
    virtual ~LineSegment2D();
};

class Arithmetic_Export LineSegment3D : public IShape
{
public:
    Point3 _pt[2];

public:
    LineSegment3D();
    LineSegment3D(const Point3& pt0 , const Point3& pt1);
    virtual ~LineSegment3D();
};

MED_IMG_END_NAMESPACE

#endif