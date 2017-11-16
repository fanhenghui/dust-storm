#ifndef MED_IMG_MI_DB_OPERATION_QUERY_DICOM_H
#define MED_IMG_MI_DB_OPERATION_QUERY_DICOM_H

#include "mi_db_operation.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryDICOM: public DBOperation {
public:
    DBOpQueryDICOM();
    virtual ~DBOpQueryDICOM();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpQueryDICOM>(new DBOpQueryDICOM());
    }
};

MED_IMG_END_NAMESPACE

#endif