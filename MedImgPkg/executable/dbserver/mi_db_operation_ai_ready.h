#ifndef MEDIMG_DB_MI_DB_OPERATION_AI_READY_H
#define MEDIMG_DB_MI_DB_OPERATION_AI_READY_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpAIReady: public IOperation {
public:
    DBOpAIReady();
    virtual ~DBOpAIReady();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpAIReady>(new DBOpAIReady());
    }
};

MED_IMG_END_NAMESPACE

#endif