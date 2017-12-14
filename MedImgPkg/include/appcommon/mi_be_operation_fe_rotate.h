#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ROTATE_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ROTATE_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFERotate : public IOperation {
public:
    BEOpFERotate();
    virtual ~BEOpFERotate();

    virtual int execute();

    CREATE_EXTENDS_OP(BEOpFERotate)
};

MED_IMG_END_NAMESPACE

#endif