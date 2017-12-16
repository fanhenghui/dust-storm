#ifndef MEDIMG_AI_MI_AI_OPERATION_DB_REQUEST_EVALUATION_H
#define MEDIMG_AI_MI_AI_OPERATION_DB_REQUEST_EVALUATION_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AIOpDBRequestEvaluation: public IOperation {
public:
    AIOpDBRequestEvaluation();
    virtual ~AIOpDBRequestEvaluation();

    virtual int execute();

    CREATE_EXTENDS_OP(AIOpDBRequestEvaluation)
};

MED_IMG_END_NAMESPACE

#endif