#ifndef MED_IMAGING_PLANE_H
#define MED_IMAGING_PLANE_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

class LineSegment;
//////////////////////////////////////////////////////////////////////////
/// \ plane function a*x + b*y + c*z = d
/// \ point& normal x*N = d (d is the distance between original and plane)
//////////////////////////////////////////////////////////////////////////
class Arithmetic_Export Plane : public IShape
{
public:
    Vector3 _norm;
    double _distance;

public:
    Plane();

    virtual ~Plane();

    double distance_to_point(const Point3 &pt) const;
};

MED_IMAGING_END_NAMESPACE

#endif