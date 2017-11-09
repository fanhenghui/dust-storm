#ifndef MED_IMG_MI_DB_OPERATION_QUERY_PROCESS_MASK_H
#define MED_IMG_MI_DB_OPERATION_QUERY_PROCESS_MASK_H

#include "mi_db_operation.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryPreprocessMask: public DBOperation {
public:
    DBOpQueryPreprocessMask();
    virtual ~DBOpQueryPreprocessMask();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpQueryPreprocessMask>(new DBOpQueryPreprocessMask());
    }
};

MED_IMG_END_NAMESPACE

#endif