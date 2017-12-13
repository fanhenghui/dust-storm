#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_AI_EVALUATION_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_AI_EVALUATION_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEFetchAIEvaluation: public IOperation {
public:
    DBOpBEFetchAIEvaluation();
    virtual ~DBOpBEFetchAIEvaluation();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpBEFetchAIEvaluation)
};

MED_IMG_END_NAMESPACE

#endif