#ifndef MED_IMG_APPCOMMON_MI_OPERATION_SWITCH_PRESET_VRT_H
#define MED_IMG_APPCOMMON_MI_OPERATION_SWITCH_PRESET_VRT_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpSwitchPresetVRT : public IOperation {
public:
    OpSwitchPresetVRT();
    virtual ~OpSwitchPresetVRT();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif