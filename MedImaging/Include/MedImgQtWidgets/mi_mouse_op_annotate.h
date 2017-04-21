#ifndef MED_IMAGING_MOUSE_OP_ANAOTATE_H_
#define MED_IMAGING_MOUSE_OP_ANAOTATE_H_

#include "MedImgQtWidgets/mi_mouse_op_interface.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE

class VOIModel;

class QtWidgets_Export MouseOpAnnotate : public IMouseOp
{
public:
    MouseOpAnnotate();
    virtual ~MouseOpAnnotate();
    virtual void press(const QPoint& pt);
    virtual void move(const QPoint& pt);
    virtual void release(const QPoint& pt);
    virtual void double_click(const QPoint& pt);
    void set_type();//TODO type 
    void set_voi_model(std::shared_ptr<VOIModel> pVOIModel);
protected:
private:
    bool m_bPin;
    medical_imaging::Point3 m_ptCenter;
    double diameter;
    std::shared_ptr<VOIModel> m_pVOIModel;
};

MED_IMAGING_END_NAMESPACE

#endif