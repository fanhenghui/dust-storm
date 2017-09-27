#ifndef MED_IMG_APPCOMMON_MI_OPERATION_WINDOWING_H
#define MED_IMG_APPCOMMON_MI_OPERATION_WINDOWING_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class OpWindowing : public IOperation {
public:
    OpWindowing();
    virtual ~OpWindowing();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif