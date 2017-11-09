#ifndef MED_IMG_MI_DB_OPERATION_QUERY_DICOM_H
#define MED_IMG_MI_DB_OPERATION_QUERY_DICOM_H

#include "mi_db_operation.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryDCM: public DBOperation {
public:
    DBOpQueryDCM();
    virtual ~DBOpQueryDCM();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpQueryDCM>(new DBOpQueryDCM());
    }
};

MED_IMG_END_NAMESPACE

#endif