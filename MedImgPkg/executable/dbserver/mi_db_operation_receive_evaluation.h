#ifndef MED_IMG_MI_DB_OPERATION_RECEIVE_INFERENCE_H
#define MED_IMG_MI_DB_OPERATION_RECEIVE_INFERENCE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpReceiveEvaluation: public IOperation {
public:
    DBOpReceiveEvaluation();
    virtual ~DBOpReceiveEvaluation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpReceiveEvaluation>(new DBOpReceiveEvaluation());
    }
};

MED_IMG_END_NAMESPACE

#endif