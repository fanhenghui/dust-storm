#ifndef MED_IMG_OPERATION_ANNOTATION_H
#define MED_IMG_OPERATION_ANNOTATION_H

#include "appcommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE

class OpAnnotation : public IOperation {
public:
    OpAnnotation();
    virtual ~OpAnnotation();

    virtual int execute();
};
MED_IMG_END_NAMESPACE

#endif