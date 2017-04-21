#ifndef MED_IMAGING_VOI_H_
#define MED_IMAGING_VOI_H_

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

struct VOISphere
{
    Point3 m_ptCenter;
    double m_dDiameter;
    std::string m_sName;

    VOISphere():m_ptCenter(Point3::S_ZERO_POINT),m_dDiameter(0),m_sName("")
    {}

    VOISphere(const Point3& pt , double d):m_ptCenter(pt),m_dDiameter(d),m_sName("")
    {}

    VOISphere(const Point3& pt , double d , const std::string& sName):m_ptCenter(pt),m_dDiameter(d),m_sName(sName)
    {}
};


MED_IMAGING_END_NAMESPACE


#endif