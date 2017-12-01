#ifndef MEDIMGIO_VOI_H
#define MEDIMGIO_VOI_H

#include "arithmetic/mi_point3.h"
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

struct VOISphere {
    Point3 center;
    double diameter;
    std::string name;
    //reserverd
    //1 means probability in lung nodule
    float para0;

    VOISphere() : center(Point3::S_ZERO_POINT), diameter(0), name(""), para0(0) {}

    VOISphere(const Point3& pt, double d) : center(pt), diameter(d), name(""), para0(0) {}

    VOISphere(const Point3& pt, double d, const std::string& name)
        : center(pt), diameter(d), name(name), para0(0) {}

    bool operator==(const VOISphere& voi) const {
        return voi.center == center &&
               fabs(voi.diameter - diameter) < DOUBLE_EPSILON;
    }

    bool operator!=(const VOISphere& voi) const {
        return fabs(voi.diameter - diameter) > DOUBLE_EPSILON ||
               voi.center != center;
    }
};

MED_IMG_END_NAMESPACE

#endif