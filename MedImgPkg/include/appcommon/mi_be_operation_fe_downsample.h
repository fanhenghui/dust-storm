#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_DOWNSAMPLE_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_DOWNSAMPLE_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEDownsample : public IOperation {
public:
    BEOpFEDownsample();
    virtual ~BEOpFEDownsample();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFEDownsample>(new BEOpFEDownsample());
    }
};
MED_IMG_END_NAMESPACE

#endif