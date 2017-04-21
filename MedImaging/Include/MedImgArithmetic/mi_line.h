#ifndef MED_IMAGING_LINE_H
#define MED_IMAGING_LINE_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_vector2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export Line2D : public IShape
{
public:
    Point2 _pt;
    Vector2 _dir;

public:
    Line2D();
    virtual ~Line2D();
};

class Arithmetic_Export Line3D : public IShape
{
public:
    Point3 _pt;
    Vector3 _dir;

public:
    Line3D();
    virtual ~Line3D();
};

MED_IMAGING_END_NAMESPACE

#endif