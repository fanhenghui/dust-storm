#ifndef MED_IMG_MOUSE_OP_ZOOM_H
#define MED_IMG_MOUSE_OP_ZOOM_H

#include "qtpackage/mi_mouse_op_interface.h"

MED_IMG_BEGIN_NAMESPACE

class QtPackage_Export MouseOpZoom : public IMouseOp
{
public:
    MouseOpZoom();
    virtual ~MouseOpZoom();

    virtual void press(const QPointF& pt);
    virtual void move(const QPointF& pt);
    virtual void release(const QPointF& pt);
    virtual void double_click(const QPointF& pt);
    virtual void wheel_slide(int);
protected:
private:
};

MED_IMG_END_NAMESPACE

#endif