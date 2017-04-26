#ifndef MED_IMAGING_MOUSE_OP_ZOOM_H
#define MED_IMAGING_MOUSE_OP_ZOOM_H

#include "MedImgQtWidgets/mi_mouse_op_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export MouseOpZoom : public IMouseOp
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

MED_IMAGING_END_NAMESPACE

#endif