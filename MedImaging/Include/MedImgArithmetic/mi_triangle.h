#ifndef MED_IMAGING_TRIANGLE_H
#define MED_IMAGING_TRIANGLE_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_line_segment.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export Triangle : public IShape
{
public:
    Point3 m_pt[3];

public:
    Triangle();

    Triangle(Point3 pt0 , Point3 pt1 ,Point3 pt2);

    void GenerateLines(std::vector<LineSegment3D> &vLines) const;

    Vector3 GetNormal() const;

    bool InTriangle(const Point3 &pt0);
};

MED_IMAGING_END_NAMESPACE
#endif