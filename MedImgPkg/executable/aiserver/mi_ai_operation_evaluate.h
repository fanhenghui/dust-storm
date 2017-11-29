#ifndef MED_IMG_MI_AI_OPERATION_INFERENCE_H
#define MED_IMG_MI_AI_OPERATION_INFERENCE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AIOpEvaluate: public IOperation {
public:
    AIOpEvaluate();
    virtual ~AIOpEvaluate();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<AIOpEvaluate>(new AIOpEvaluate());
    }
};

MED_IMG_END_NAMESPACE

#endif