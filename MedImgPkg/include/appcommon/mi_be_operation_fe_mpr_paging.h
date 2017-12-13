#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_MPR_PAGING_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_MPR_PAGING_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEMPRPaging : public IOperation {
public:
    BEOpFEMPRPaging();
    virtual ~BEOpFEMPRPaging();

    virtual int execute();

    CREATE_EXTENDS_OP(BEOpFEMPRPaging)
};

MED_IMG_END_NAMESPACE

#endif