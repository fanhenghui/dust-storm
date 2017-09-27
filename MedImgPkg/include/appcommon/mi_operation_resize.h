#ifndef MED_IMG_APPCOMMON_MI_OPERATION_RESIZE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_RESIZE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class OpResize : public IOperation {
public:
    OpResize();
    virtual ~OpResize();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif