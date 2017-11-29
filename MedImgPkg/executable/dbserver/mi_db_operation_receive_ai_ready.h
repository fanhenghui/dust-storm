#ifndef MED_IMG_MI_DB_OPERATION_AIS_READY_H
#define MED_IMG_MI_DB_OPERATION_AIS_READY_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpReceiveAIReady: public IOperation {
public:
    DBOpReceiveAIReady();
    virtual ~DBOpReceiveAIReady();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpReceiveAIReady>(new DBOpReceiveAIReady());
    }
};

MED_IMG_END_NAMESPACE

#endif