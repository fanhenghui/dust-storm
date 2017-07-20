#ifndef MED_IMAGING_MOUSE_OP_ANAOTATE_FINE_TUNING_H_
#define MED_IMAGING_MOUSE_OP_ANAOTATE_FINE_TUNING_H_

#include "MedImgQtWidgets/mi_mouse_op_interface.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

class VOIModel;

class QtWidgets_Export MouseOpAnnotateFineTuning : public IMouseOp
{
public:
    MouseOpAnnotateFineTuning();
    virtual ~MouseOpAnnotateFineTuning();

    virtual void press(const QPointF& pt);
    virtual void move(const QPointF& pt);
    virtual void release(const QPointF& pt);
    virtual void double_click(const QPointF& pt);
    virtual void wheel_slide(int value);

    void set_voi_model(std::shared_ptr<VOIModel> model);

    enum TuneType {
        SUBSTRACT = -1,
        ADD = +1,
    };
    void set_tune_type(const int new_tune_type)
    {
        _tune_type = new_tune_type;
    };

    enum ShapeType {
        SPHERE = 0,
        CUBE = 1,
        SHAPETYPE_COUNT,
    };
    void set_shape_type(const int new_shape_type)
    {
        _shape_type = new_shape_type;
    };

protected:

private:
    int _tune_status;
    int _tune_type;
    int _shape_type;
    std::shared_ptr<VOIModel> _model;
};

MED_IMAGING_END_NAMESPACE

#endif