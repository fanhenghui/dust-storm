#ifndef MEDIMGARITHMETIC_MI_ELLIPSOID_H
#define MEDIMGARITHMETIC_MI_ELLIPSOID_H

#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_shape_interface.h"
#include "arithmetic/mi_vector3.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Ellipsoid : public IShape {
public:
    Point3 _center;
    double _a, _b, _c;

public:
    Ellipsoid():_center(Point3::S_ZERO_POINT), _a(1), _b(1), _c(1)  {}
    virtual ~Ellipsoid() {}

    inline bool in_ellipsoid(const Point3& pt) {
        Vector3 tmp = pt - _center;

        if (!(tmp.x * tmp.x / _a * _a + tmp.y * tmp.y / _b * _b +
                tmp.z * tmp.z / _c * _c >
                1.0)) {
            return true;
        } else {
            return false;
        }
    }
};

MED_IMG_END_NAMESPACE

#endif