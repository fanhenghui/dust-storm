#ifndef MEDIMGARITHMETIC_MI_PLANE_H
#define MEDIMGARITHMETIC_MI_PLANE_H

#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_shape_interface.h"

MED_IMG_BEGIN_NAMESPACE

class LineSegment;
//////////////////////////////////////////////////////////////////////////
/// \ plane function a*x + b*y + c*z = d
/// \ point& normal x*N = d (d is the distance between original and plane)
//////////////////////////////////////////////////////////////////////////
class Arithmetic_Export Plane : public IShape {
public:
    Vector3 _norm;
    double _distance;

public:
    Plane();

    virtual ~Plane();

    double distance_to_point(const Point3& pt) const;
};

MED_IMG_END_NAMESPACE

#endif