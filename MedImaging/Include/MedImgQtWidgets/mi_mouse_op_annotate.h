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
    virtual void Press(const QPoint& pt);
    virtual void Move(const QPoint& pt);
    virtual void Release(const QPoint& pt);
    virtual void DoubleClick(const QPoint& pt);
    void SetType();//TODO type 
    void SetVOIModel(std::shared_ptr<VOIModel> pVOIModel);
protected:
private:
    bool m_bPin;
    MedImaging::Point3 m_ptCenter;
    double m_dDiameter;
    std::shared_ptr<VOIModel> m_pVOIModel;
};

MED_IMAGING_END_NAMESPACE

#endif