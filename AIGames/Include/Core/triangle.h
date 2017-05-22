#ifndef MED_IMAGING_TRIANGLE_H
#define MED_IMAGING_TRIANGLE_H

#include "Core/shape_interface.h"
#include "Core/point2.h"
#include "Core/point3.h"
#include "Core/line_segment.h"

class  Triangle : public IShape
{
public:
    Point3 _pt[3];

public:
    Triangle();

    Triangle(Point3 pt0 , Point3 pt1 ,Point3 pt2);

    void to_lines(std::vector<LineSegment3D> &lines) const;

    Vector3 get_normal() const;

    bool in_triangle(const Point3 &pt0);
};


#endif