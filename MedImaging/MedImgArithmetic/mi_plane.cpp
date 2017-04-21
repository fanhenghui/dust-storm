#include "mi_plane.h"

MED_IMAGING_BEGIN_NAMESPACE

Plane::Plane()
{

}

Plane::~Plane()
{

}

double Plane::distance_to_point(const Point3 &pt) const
{
     return _norm.dot_product(pt - Point3::S_ZERO_POINT) - _distance;
}

MED_IMAGING_END_NAMESPACE