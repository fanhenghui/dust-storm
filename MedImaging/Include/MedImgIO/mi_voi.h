#ifndef MED_IMAGING_VOI_H_
#define MED_IMAGING_VOI_H_

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

struct VOISphere
{
    Point3 center;
    double diameter;
    std::string name;

    VOISphere():center(Point3::S_ZERO_POINT),diameter(0),name("")
    {}

    VOISphere(const Point3& pt , double d):center(pt),diameter(d),name("")
    {}

    VOISphere(const Point3& pt , double d , const std::string& name):center(pt),diameter(d),name(name)
    {}

    bool operator == (const VOISphere& voi)
    {
        return voi.center == center && abs(voi.diameter - diameter) < DOUBLE_EPSILON;
    }
};

bool IO_Export operator == (const VOISphere& l , const VOISphere& r);


MED_IMAGING_END_NAMESPACE


#endif