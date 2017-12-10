#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ANNOTATION_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ANNOTATION_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BNOpFEAnnotation : public IOperation {
public:
    BNOpFEAnnotation();
    virtual ~BNOpFEAnnotation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BNOpFEAnnotation>(new BNOpFEAnnotation());
    }
};
MED_IMG_END_NAMESPACE

#endif