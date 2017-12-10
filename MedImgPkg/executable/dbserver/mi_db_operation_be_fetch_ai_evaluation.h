#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_AI_EVALUATION_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_AI_EVALUATION_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEFetchAIEvaluation: public IOperation {
public:
    DBOpBEFetchAIEvaluation();
    virtual ~DBOpBEFetchAIEvaluation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpBEFetchAIEvaluation>(new DBOpBEFetchAIEvaluation());
    }
};

MED_IMG_END_NAMESPACE

#endif