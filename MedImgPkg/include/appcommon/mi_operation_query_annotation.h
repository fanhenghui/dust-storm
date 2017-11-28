#ifndef MED_IMG_APPCOMMON_MI_OPERATION_QUERY_ANNOTATION_H
#define MED_IMG_APPCOMMON_MI_OPERATION_QUERY_ANNOTATION_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpQueryAnnotation : public IOperation {
public:
    OpQueryAnnotation();
    virtual ~OpQueryAnnotation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpQueryAnnotation>(new OpQueryAnnotation());
    }
};
MED_IMG_END_NAMESPACE

#endif