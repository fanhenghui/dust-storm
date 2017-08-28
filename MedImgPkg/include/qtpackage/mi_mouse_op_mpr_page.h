#ifndef MED_IMG_MOUSE_OP_MPR_PAGING_H
#define MED_IMG_MOUSE_OP_MPR_PAGING_H

#include "qtpackage/mi_mouse_op_interface.h"

MED_IMG_BEGIN_NAMESPACE
class CrosshairModel;
class QtPackage_Export MouseOpMPRPage : public IMouseOp
{
public:
    MouseOpMPRPage();
    virtual ~MouseOpMPRPage();

    virtual void press(const QPointF& pt);
    virtual void move(const QPointF& pt);
    virtual void release(const QPointF& pt);
    virtual void double_click(const QPointF& pt);
    virtual void wheel_slide(int);

    void set_crosshair_model(std::shared_ptr<CrosshairModel> model);
protected:
private:
    std::shared_ptr<CrosshairModel> _model;
};

MED_IMG_END_NAMESPACE

#endif