#ifndef MED_IMAGING_MOUSE_OP_TEST_H
#define MED_IMAGING_MOUSE_OP_TEST_H

#include "qtpackage/mi_mouse_op_interface.h"

MED_IMG_BEGIN_NAMESPACE

class QtWidgets_Export MouseOpTest : public IMouseOp
{
public:
    MouseOpTest();
    virtual ~MouseOpTest();

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