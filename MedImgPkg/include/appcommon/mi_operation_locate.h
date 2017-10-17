#ifndef MED_IMG_APPCOMMON_MI_OPERATION_LOCATE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_LOCATE_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpLocate : public IOperation {
public:
    OpLocate();
    virtual ~OpLocate();

    virtual int execute();
};
MED_IMG_END_NAMESPACE

#endif