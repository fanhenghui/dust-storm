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
     return m_vNorm.dot_product(pt - Point3::kZeroPoint) - m_dDistance;
}

MED_IMAGING_END_NAMESPACE