#include "intersection_test.h"


bool IntersectionTest::plane_to_plane(const Plane& plane1,const Plane& plane2,Line3D& intersected_line)
{
    const Vector3 norm1 = plane1._norm;//Here should be normalized
    const Vector3 norm2 = plane2._norm;
    if( norm1 == norm2)
    {
        return false;
    }
    else
    {
        Vector3 vDir = plane1._norm.cross_product(plane2._norm);
        vDir.normalize();
        intersected_line._dir = vDir;
        const double a0 = plane1._norm.x;
        const double b0 = plane1._norm.y;
        const double c0 = plane1._norm.z;
        const double d0 = plane1._distance;

        const double a1 = plane2._norm.x;
        const double b1 = plane2._norm.y;
        const double c1 = plane2._norm.z;
        const double d1 = plane2._distance;

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

        intersected_line._pt = Point3(x,y,z);
        return true;
    }
}

