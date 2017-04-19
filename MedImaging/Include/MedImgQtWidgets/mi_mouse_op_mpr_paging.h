#ifndef MED_IMAGING_MOUSE_OP_MPR_PAGING_H
#define MED_IMAGING_MOUSE_OP_MPR_PAGING_H

#include "MedImgQtWidgets/mi_mouse_op_interface.h"

MED_IMAGING_BEGIN_NAMESPACE
class CrosshairModel;
class QtWidgets_Export MouseOpMPRPaging : public IMouseOp
{
public:
    MouseOpMPRPaging();
    virtual ~MouseOpMPRPaging();
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