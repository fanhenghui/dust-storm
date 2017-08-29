#ifndef MEDIMGARITHMETIC_MI_TRIANGLE_H
#define MEDIMGARITHMETIC_MI_TRIANGLE_H

#include "arithmetic/mi_line_segment.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_shape_interface.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Triangle : public IShape {
public:
    Point3 _pt[3];

public:
    Triangle() {
        _pt[0] = Point3::S_ZERO_POINT;
        _pt[1] = Point3::S_ZERO_POINT;
        _pt[2] = Point3::S_ZERO_POINT;
    }

    Triangle(Point3 pt0, Point3 pt1, Point3 pt2) {
        _pt[0] = pt0;
        _pt[1] = pt1;
        _pt[2] = pt2;
    }

    void to_lines(std::vector<LineSegment3D>& lines) const;

    Vector3 get_normal() const;

    bool in_triangle(const Point3& pt0);
};

MED_IMG_END_NAMESPACE
#endif