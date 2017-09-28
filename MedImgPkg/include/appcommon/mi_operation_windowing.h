#ifndef MED_IMG_APPCOMMON_MI_OPERATION_WINDOWING_H
#define MED_IMG_APPCOMMON_MI_OPERATION_WINDOWING_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpWindowing : public IOperation {
public:
    OpWindowing();
    virtual ~OpWindowing();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif