#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_PAN_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_PAN_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEPan : public IOperation {
public:
    BEOpFEPan();
    virtual ~BEOpFEPan();

    virtual int execute();

    CREATE_EXTENDS_OP(BEOpFEPan)
};

MED_IMG_END_NAMESPACE

#endif