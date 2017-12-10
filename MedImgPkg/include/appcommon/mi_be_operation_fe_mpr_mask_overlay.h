#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_MPR_MASK_OVERLAY_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_MPR_MASK_OVERLAY_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEMPRMaskOverlay : public IOperation {
public:
    BEOpFEMPRMaskOverlay();
    virtual ~BEOpFEMPRMaskOverlay();

    virtual int execute();
    
    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFEMPRMaskOverlay>(new BEOpFEMPRMaskOverlay());
    }

private:
};

MED_IMG_END_NAMESPACE

#endif