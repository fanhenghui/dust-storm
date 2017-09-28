#ifndef MEDIMGARITHMETIC_MI_CIRCLE_H
#define MEDIMGARITHMETIC_MI_CIRCLE_H

#include "arithmetic/mi_arithmetic_export.h"
#include "arithmetic/mi_point2.h"

MED_IMG_BEGIN_NAMESPACE

class Circle {
public:
    Circle() : _center(Point2::S_ZERO_POINT), _radius(0) {};
    Circle(Point2 c, double r) : _center(c), _radius(r) {};

    inline bool operator == (const Circle& circle) {
        return(circle._center == _center && fabs(circle._radius - _radius) < FLOAT_EPSILON);
    }

    inline bool operator != (const Circle& circle) {
        return(circle._center != _center || fabs(circle._radius - _radius) > FLOAT_EPSILON);
    }

public:
    Point2 _center;
    double _radius;
};

MED_IMG_END_NAMESPACE

#endif