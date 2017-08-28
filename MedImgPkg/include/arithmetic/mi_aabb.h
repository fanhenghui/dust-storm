#ifndef MED_IMG_AABB_H
#define MED_IMG_AABB_H

#include "arithmetic/mi_shape_interface.h"
#include "arithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE 

class Arithmetic_Export AABB : public IShape
{
public:
    Point3 _min;//Lower Left Back
    Point3 _max;//Upper Right Front

public:
    AABB();
    virtual ~AABB();

    bool operator == (const AABB& aabb) const;
    bool operator != (const AABB& aabb) const;
};

class Arithmetic_Export AABBUI : public IShape
{
public:
    unsigned int _min[3];//Lower Left Back
    unsigned int _max[3];//Upper Right Front

public:
    AABBUI();
    AABBUI(const unsigned int (&min0)[3] , const unsigned int (&max0)[3]);
    virtual ~AABBUI();

    bool operator == (const AABBUI& aabb) const;
    bool operator != (const AABBUI& aabb) const;

    void Print();
};

class Arithmetic_Export AABBI : public IShape
{
public:
    int _min[3];//Lower Left Back
    int _max[3];//Upper Right Front

public:
    AABBI();
    AABBI(const int (&min0)[3] , const int (&max0)[3]);
    virtual ~AABBI();

    bool operator == (const AABBI& aabb) const;
    bool operator != (const AABBI& aabb) const;

    void Print();
};

MED_IMG_END_NAMESPACE

#endif