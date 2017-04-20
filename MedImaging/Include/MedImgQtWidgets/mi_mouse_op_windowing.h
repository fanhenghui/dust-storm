#ifndef MED_IMAGING_MOUSE_OP_WINDOWING_H
#define MED_IMAGING_MOUSE_OP_WINDOWING_H

#include "MedImgQtWidgets/mi_mouse_op_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export MouseOpWindowing : public IMouseOp
{
public:
    MouseOpWindowing();
    virtual ~MouseOpWindowing();
    virtual void press(const QPoint& pt);
    virtual void move(const QPoint& pt);
    virtual void release(const QPoint& pt);
    virtual void double_click(const QPoint& pt);
protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif