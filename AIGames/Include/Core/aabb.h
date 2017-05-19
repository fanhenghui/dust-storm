#ifndef MED_IMAGING_AABB_H
#define MED_IMAGING_AABB_H

#include "Core/shape_interface.h"
#include "Core/point3.h"

class  AABB : public IShape
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

class  AABBUI : public IShape
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



#endif