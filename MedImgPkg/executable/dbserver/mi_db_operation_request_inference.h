#ifndef MED_IMG_MI_DB_OPERATION_REQUEST_INFERENCE_H
#define MED_IMG_MI_DB_OPERATION_REQUEST_INFERENCE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpRequestInference: public IOperation {
public:
    DBOpRequestInference();
    virtual ~DBOpRequestInference();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpRequestInference>(new DBOpRequestInference());
    }
};

MED_IMG_END_NAMESPACE

#endif