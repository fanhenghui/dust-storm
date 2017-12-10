#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_SWITCH_PRESET_VRT_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_SWITCH_PRESET_VRT_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFESwitchPresetVRT : public IOperation {
public:
    BEOpFESwitchPresetVRT();
    virtual ~BEOpFESwitchPresetVRT();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFESwitchPresetVRT>(new BEOpFESwitchPresetVRT());
    }

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif