#ifndef MED_IMG_APPCOMMON_MI_OPERATION_ANNOTATION_H
#define MED_IMG_APPCOMMON_MI_OPERATION_ANNOTATION_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpAnnotation : public IOperation {
public:
    OpAnnotation();
    virtual ~OpAnnotation();

    virtual int execute();
};
MED_IMG_END_NAMESPACE

#endif