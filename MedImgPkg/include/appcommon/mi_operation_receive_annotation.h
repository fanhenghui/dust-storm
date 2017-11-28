#ifndef MED_IMG_APPCOMMON_MI_OPERATION_RECEIVE_ANNOTATION_H
#define MED_IMG_APPCOMMON_MI_OPERATION_RECEIVE_ANNOTATION_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpReceiveAnnotation : public IOperation {
public:
    OpReceiveAnnotation();
    virtual ~OpReceiveAnnotation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpReceiveAnnotation>(new OpReceiveAnnotation());
    }
};
MED_IMG_END_NAMESPACE

#endif