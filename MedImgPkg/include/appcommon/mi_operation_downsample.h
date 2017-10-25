#ifndef MED_IMG_APPCOMMON_MI_OPERATION_DOWNSAMPLE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_DOWNSAMPLE_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpDownsample : public IOperation {
public:
    OpDownsample();
    virtual ~OpDownsample();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpDownsample>(new OpDownsample());
    }
};
MED_IMG_END_NAMESPACE

#endif