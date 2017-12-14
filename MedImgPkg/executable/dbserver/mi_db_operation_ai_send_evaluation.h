#ifndef MEDIMG_DB_MI_DB_OPERATION_AI_SEND_EVALUATION_H
#define MEDIMG_DB_MI_DB_OPERATION_AI_SEND_EVALUATION_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpAISendEvaluation: public IOperation {
public:
    DBOpAISendEvaluation();
    virtual ~DBOpAISendEvaluation();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpAISendEvaluation)
};

MED_IMG_END_NAMESPACE

#endif