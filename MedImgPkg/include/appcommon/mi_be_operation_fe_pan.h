#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_PAN_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_PAN_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEPan : public IOperation {
public:
    BEOpFEPan();
    virtual ~BEOpFEPan();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFEPan>(new BEOpFEPan());
    }

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif