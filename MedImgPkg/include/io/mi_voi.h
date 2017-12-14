#ifndef MEDIMGIO_VOI_H
#define MEDIMGIO_VOI_H

#include "arithmetic/mi_point3.h"
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

struct VOISphere {
    Point3 center;
    double diameter;
    std::string name;
    float probability;

    VOISphere() : center(Point3::S_ZERO_POINT), diameter(0), name(""), probability(0) {}

    VOISphere(const Point3& pt, double d) : center(pt), diameter(d), name(""), probability(0) {}

    VOISphere(const Point3& pt, double d, const std::string& name)
        : center(pt), diameter(d), name(name), probability(0) {}

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