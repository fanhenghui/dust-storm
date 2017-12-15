#ifndef MEDIMG_AI_MI_AI_OPERATION_DB_REQUEST_EVALUATION_H
#define MEDIMG_AI_MI_AI_OPERATION_DB_REQUEST_EVALUATION_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AIODBRequestEvaluation: public IOperation {
public:
    AIODBRequestEvaluation();
    virtual ~AIODBRequestEvaluation();

    virtual int execute();

    CREATE_EXTENDS_OP(AIODBRequestEvaluation)
};

MED_IMG_END_NAMESPACE

#endif