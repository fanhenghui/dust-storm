#ifndef MED_IMAGING_MOUSE_OP_MPR_PAGING_H
#define MED_IMAGING_MOUSE_OP_MPR_PAGING_H

#include "MedImgQtWidgets/mi_mouse_op_interface.h"

MED_IMAGING_BEGIN_NAMESPACE
class CrosshairModel;
class QtWidgets_Export MouseOpMPRPage : public IMouseOp
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

MED_IMAGING_END_NAMESPACE

#endif