#ifndef MED_IMG_MI_DB_OPERATION_RECEIVE_INFERENCE_H
#define MED_IMG_MI_DB_OPERATION_RECEIVE_INFERENCE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpReceiveInference: public IOperation {
public:
    DBOpReceiveInference();
    virtual ~DBOpReceiveInference();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpReceiveInference>(new DBOpReceiveInference());
    }
};

MED_IMG_END_NAMESPACE

#endif