#ifndef MED_IMG_MI_DB_OPERATION_FETCH_AI_EVALUATION_H
#define MED_IMG_MI_DB_OPERATION_FETCH_AI_EVALUATION_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpFetchAIEvaluation: public IOperation {
public:
    DBOpFetchAIEvaluation();
    virtual ~DBOpFetchAIEvaluation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpFetchAIEvaluation>(new DBOpFetchAIEvaluation());
    }
};

MED_IMG_END_NAMESPACE

#endif