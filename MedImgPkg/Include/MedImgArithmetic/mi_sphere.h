#ifndef MED_IMG_SPHERE_H
#define MED_IMG_SPHERE_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Sphere : public IShape
{
public:
    Point3 _center;
    double _radius;

public:
    Sphere();
    virtual ~Sphere();
};
MED_IMG_END_NAMESPACE

#endif