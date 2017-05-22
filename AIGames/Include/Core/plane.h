#ifndef MED_IMAGING_PLANE_H
#define MED_IMAGING_PLANE_H

#include "Core/shape_interface.h"
#include "Core/point2.h"
#include "Core/point3.h"

class LineSegment;
//////////////////////////////////////////////////////////////////////////
/// \ plane function a*x + b*y + c*z = d
/// \ point& normal x*N = d (d is the distance between original and plane)
//////////////////////////////////////////////////////////////////////////
class  Plane : public IShape
{
public:
    Vector3 _norm;
    double _distance;

public:
    Plane();

    virtual ~Plane();

    double distance_to_point(const Point3 &pt) const;
};



#endif