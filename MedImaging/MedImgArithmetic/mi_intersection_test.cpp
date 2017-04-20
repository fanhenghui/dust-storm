#include "mi_intersection_test.h"
#include "mi_vector2f.h"

MED_IMAGING_BEGIN_NAMESPACE

bool IntersectionTest::plane_to_plane(const Plane& plane1,const Plane& plane2,Line3D& intersectingLine)
{
    const Vector3 vNorm1 = plane1.m_vNorm;//Here should be normalised
    const Vector3 vNorm2 = plane2.m_vNorm;
    if( vNorm1 == vNorm2)
    {
        return false;
    }
    else
    {
        Vector3 vDir = plane1.m_vNorm.cross_product(plane2.m_vNorm);
        vDir.normalize();
        intersectingLine.m_vDir = vDir;
        const double a0 = plane1.m_vNorm.x;
        const double b0 = plane1.m_vNorm.y;
        const double c0 = plane1.m_vNorm.z;
        const double d0 = plane1.m_dDistance;

        const double a1 = plane2.m_vNorm.x;
        const double b1 = plane2.m_vNorm.y;
        const double c1 = plane2.m_vNorm.z;
        const double d1 = plane2.m_dDistance;

        double x,y,z;
        if (std::abs(a0*b1 - a1*b0) > DOUBLE_EPSILON)
        {
            x = (b1*d0 - b0*d1)/(a0*b1 - a1*b0);
            y = (a1*d0 - a0*d1)/(a1*b0 - a0*b1);
            z = 0;
        }
        else if (std::abs(c0*b1 - c1*b0) > DOUBLE_EPSILON)
        {
            x = 0;
            y = (c1*d0 - c0*d1)/(c1*b0 - c0*b1);
            z = (b1*d0 - b0*d1)/(c0*b1 - c1*b0);
        }
        else if (std::abs(a0*c1 - a1*c0) > DOUBLE_EPSILON)
        {
            x = (c1*d0 - c0*d1)/(a0*c1 - a1*c0);
            y = 0;
            z = (a1*d0 - a0*d1)/(a1*c0 - a0*c1);
        }
        else
        {
            return false;
        }

        intersectingLine.m_pt = Point3(x,y,z);
        return true;
    }
}

MED_IMAGING_END_NAMESPACE