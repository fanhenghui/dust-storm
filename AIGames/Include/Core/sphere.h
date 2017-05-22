#ifndef MED_IMAGING_SPHERE_H
#define MED_IMAGING_SPHERE_H

#include "Core/shape_interface.h"
#include "Core/point2.h"
#include "Core/point3.h"

class  Sphere : public IShape
{
public:
    Point3 _center;
    double _radius;

public:
    Sphere();
    virtual ~Sphere();
};


#endif