#ifndef MED_IMAGING_LINE_H
#define MED_IMAGING_LINE_H

#include "Core/shape_interface.h"
#include "Core/point2.h"
#include "Core/vector2.h"
#include "Core/point3.h"
#include "Core/vector3.h"

class  Line2D : public IShape
{
public:
    Point2 _pt;
    Vector2 _dir;

public:
    Line2D();
    virtual ~Line2D();

    bool operator == (const Line2D& l)
    {
        return l._pt == _pt && l._dir == _dir;
    }
};

class  Line3D : public IShape
{
public:
    Point3 _pt;
    Vector3 _dir;

public:
    Line3D();
    virtual ~Line3D();
};



#endif