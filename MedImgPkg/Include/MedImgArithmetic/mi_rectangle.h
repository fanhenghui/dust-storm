#ifndef MED_IMG_RECTANGLE_H
#define MED_IMG_RECTANGLE_H

#include "MedImgArithmetic/mi_shape_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export Rectangle : public IShape
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

MED_IMG_END_NAMESPACE

#endif