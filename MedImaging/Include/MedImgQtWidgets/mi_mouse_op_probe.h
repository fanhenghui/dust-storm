#ifndef MED_IMAGING_MOUSE_OP_PROBE_H
#define MED_IMAGING_MOUSE_OP_PROBE_H

#include "MedImgQtWidgets/mi_mouse_op_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export MouseOpProbe : public IMouseOp
{
public:
    MouseOpProbe();
    virtual ~MouseOpProbe();
    virtual void Press(const QPoint& pt);
    virtual void Move(const QPoint& pt);
    virtual void Release(const QPoint& pt);
    virtual void DoubleClick(const QPoint& pt);
protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif