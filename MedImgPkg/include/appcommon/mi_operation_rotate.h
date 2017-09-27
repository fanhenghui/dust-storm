#ifndef MED_IMG_APPCOMMON_MI_OPERATION_ROTATE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_ROTATE_H

#include "appcommon/mi_operation_interface.h"
MED_IMG_BEGIN_NAMESPACE

class OpRotate : public IOperation {
public:
    OpRotate();
    virtual ~OpRotate();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif