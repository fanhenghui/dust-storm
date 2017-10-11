#ifndef MED_IMG_APPCOMMON_MI_OPERATION_MPR_MASK_OVERLAY_H
#define MED_IMG_APPCOMMON_MI_OPERATION_MPR_MASK_OVERLAY_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpMPRMaskOverlay : public IOperation {
public:
    OpMPRMaskOverlay();
    virtual ~OpMPRMaskOverlay();
    virtual int execute();

private:
};

MED_IMG_END_NAMESPACE

#endif