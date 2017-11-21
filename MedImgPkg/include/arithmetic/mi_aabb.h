#ifndef MEDIMGARITHMETIC_MI_AABB_H
#define MEDIMGARITHMETIC_MI_AABB_H

#include <ostream>
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_shape_interface.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export AABB : public IShape {
public:
    Point3 _min; // Lower Left Back
    Point3 _max; // Upper Right Front

public:
    AABB();
    virtual ~AABB();

    bool operator==(const AABB& aabb) const;
    bool operator!=(const AABB& aabb) const;

    friend std::ostream& operator << (std::ostream& strm , const AABB& aabb) {
        strm << "AABB : [ " << aabb._min.x << " " << aabb._min.y << " " << aabb._min.z
        << " ] , [" << aabb._max.x << " " << aabb._max.y << " " << aabb._max.z
        << " ]";
        return strm;
    }
};

class Arithmetic_Export AABBUI : public IShape {
public:
    unsigned int _min[3]; // Lower Left Back
    unsigned int _max[3]; // Upper Right Front

public:
    AABBUI();
    AABBUI(const unsigned int (&min0)[3], const unsigned int (&max0)[3]);
    virtual ~AABBUI();

    bool operator==(const AABBUI& aabb) const;
    bool operator!=(const AABBUI& aabb) const;

    int volume() const;

    friend std::ostream& operator << (std::ostream& strm , const AABBUI& aabb) {
        strm << "AABBUI : [ " << aabb._min[0] << " " << aabb._min[1] << " " << aabb._min[2]
        << " ] , [" << aabb._max[0] << " " << aabb._max[1] << " " << aabb._max[2]
        << " ]";
        return strm;
    }
};

class Arithmetic_Export AABBI : public IShape {
public:
    int _min[3]; // Lower Left Back
    int _max[3]; // Upper Right Front

public:
    AABBI();
    AABBI(const int (&min0)[3], const int (&max0)[3]);
    virtual ~AABBI();

    bool operator==(const AABBI& aabb) const;
    bool operator!=(const AABBI& aabb) const;

    int volume() const;

    friend std::ostream& operator << (std::ostream& strm , const AABBI& aabb) {
        strm << "AABBI : [ " << aabb._min[0] << " " << aabb._min[1] << " " << aabb._min[2]
        << " ] , [" << aabb._max[0] << " " << aabb._max[1] << " " << aabb._max[2]
        << " ]";
        return strm;
    }
};

MED_IMG_END_NAMESPACE

#endif