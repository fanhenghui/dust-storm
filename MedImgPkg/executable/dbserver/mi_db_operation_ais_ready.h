#ifndef MED_IMG_MI_DB_OPERATION_AIS_READY_H
#define MED_IMG_MI_DB_OPERATION_AIS_READY_H

#include "mi_db_operation.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpAISReady: public DBOperation {
public:
    DBOpAISReady();
    virtual ~DBOpAISReady();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpAISReady>(new DBOpAISReady());
    }
};

MED_IMG_END_NAMESPACE

#endif