#include "mi_plane.h"

MED_IMAGING_BEGIN_NAMESPACE

Plane::Plane()
{

}

Plane::~Plane()
{

}

double Plane::DistanceToPoint(const Point3 &pt) const
{
     return m_vNorm.DotProduct(pt - Point3::kZeroPoint) - m_dDistance;
}

MED_IMAGING_END_NAMESPACE