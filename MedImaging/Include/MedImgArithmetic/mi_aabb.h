#ifndef MED_IMAGING_AABB_H
#define MED_IMAGING_AABB_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export AABB : public IShape
{
public:
    Point3 m_ptLLB;//Lower Left Back
    Point3 m_ptURF;//Upper Right Front

public:
    AABB();
    virtual ~AABB();
};

MED_IMAGING_END_NAMESPACE

#endif