#ifndef MED_IMG_APPCOMMON_MI_OPERATION_SWITCH_PRESET_WINDOWING_H
#define MED_IMG_APPCOMMON_MI_OPERATION_SWITCH_PRESET_WINDOWING_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpSwitchPresetWindowing : public IOperation {
public:
    OpSwitchPresetWindowing();
    virtual ~OpSwitchPresetWindowing();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpSwitchPresetWindowing>(new OpSwitchPresetWindowing());
    }

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif