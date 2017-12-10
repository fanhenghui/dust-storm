#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_SWITCH_PRESET_WINDOWING_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_SWITCH_PRESET_WINDOWING_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFESwitchPresetWindowing : public IOperation {
public:
    BEOpFESwitchPresetWindowing();
    virtual ~BEOpFESwitchPresetWindowing();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFESwitchPresetWindowing>(new BEOpFESwitchPresetWindowing());
    }

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif