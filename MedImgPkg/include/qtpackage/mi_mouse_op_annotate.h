#ifndef MED_IMAGING_MOUSE_OP_ANAOTATE_H_
#define MED_IMAGING_MOUSE_OP_ANAOTATE_H_

#include "qtpackage/mi_mouse_op_interface.h"
#include "arithmetic/mi_point3.h"

MED_IMG_BEGIN_NAMESPACE

class VOIModel;

class QtWidgets_Export MouseOpAnnotate : public IMouseOp
{
public:
    MouseOpAnnotate();
    virtual ~MouseOpAnnotate();
    virtual void press(const QPointF& pt);
    virtual void move(const QPointF& pt);
    virtual void release(const QPointF& pt);
    virtual void double_click(const QPointF& pt);
    virtual void wheel_slide(int value);

    void set_voi_model(std::shared_ptr<VOIModel> model);
protected:
private:
    bool _is_pin;
    medical_imaging::Point3 _center;
    double _diameter;
    std::shared_ptr<VOIModel> _model;
    unsigned char _current_label;
};

MED_IMG_END_NAMESPACE

#endif