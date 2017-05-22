#ifndef MED_IMAGING_RECTANGLE_H
#define MED_IMAGING_RECTANGLE_H

#include "Core/shape_interface.h"
#include "Core/point2.h"
#include "Core/point3.h"

class  Rectangle : public IShape
{
public:
    //////////////////////////////////////////////////////////////////////////
    /// \
    //   pt[3]-----------------------------------   pt[2]
    //           |     *                                              |
    //           |              *                                     |
    //           |                       *                            |
    //           |                               *                    |
    //           |                                       *            |
    //           |                                             *      |
    //   pt[0] -----------------------------------   pt[1]
    //////////////////////////////////////////////////////////////////////////
    Point3 _pt[4];

public:
    Rectangle();

    virtual ~Rectangle();

};



#endif