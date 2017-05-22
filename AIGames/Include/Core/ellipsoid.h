#ifndef MED_IMAGING_ELLIPSOID_H
#define MED_IMAGING_ELLIPSOID_H

#include "Core/shape_interface.h"
#include "Core/point3.h"
#include "Core/vector3.h"

class  Ellipsoid: public IShape
{
public:
    Point3 _center;
    double _a , _b , _c;

public:
    Ellipsoid();
    virtual ~Ellipsoid();

    inline bool in_ellipsoid(const Point3& pt)
    {
        Vector3 tmp = pt - _center;
        if ( !(tmp.x*tmp.x / _a*_a + tmp.y*tmp.y / _b*_b + tmp.z*tmp.z / _c*_c  > 1.0) )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

};



#endif