#ifndef MED_IMAGING_MOUSE_OP_LOCATE_H_
#define MED_IMAGING_MOUSE_OP_LOCATE_H_

#include "MedImgQtWidgets/mi_mouse_op_interface.h"
#include "MedImgArithmetic/mi_point3.h"

MED_IMAGING_BEGIN_NAMESPACE
class CrosshairModel;
class QtWidgets_Export MouseOpLocate : public IMouseOp
{
public:
    MouseOpLocate();
    virtual ~MouseOpLocate();
    virtual void Press(const QPoint& pt);
    virtual void Move(const QPoint& pt);
    virtual void Release(const QPoint& pt);
    virtual void DoubleClick(const QPoint& pt);
    void SetCrosshairModel(std::shared_ptr<CrosshairModel> pModel);
protected:
private:
    std::shared_ptr<CrosshairModel> m_pModel;
};

MED_IMAGING_END_NAMESPACE

#endif