#ifndef MED_IMG_MI_DB_OPERATION_QUERY_END_H
#define MED_IMG_MI_DB_OPERATION_QUERY_END_H

#include "mi_db_operation.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryEnd: public DBOperation {
public:
    DBOpQueryEnd();
    virtual ~DBOpQueryEnd();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpQueryEnd>(new DBOpQueryEnd());
    }
};

MED_IMG_END_NAMESPACE

#endif