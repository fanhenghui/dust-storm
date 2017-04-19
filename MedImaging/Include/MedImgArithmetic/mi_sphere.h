#ifndef MED_IMAGING_SPHERE_H
#define MED_IMAGING_SPHERE_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export Sphere : public IShape
{
public:
    Point3 m_ptCenter;
    double m_dRadius;

public:
    Sphere();
    virtual ~Sphere();
};
MED_IMAGING_END_NAMESPACE

#endif