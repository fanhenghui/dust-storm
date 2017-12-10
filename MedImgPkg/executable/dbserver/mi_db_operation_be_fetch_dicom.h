#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_DICOM_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_DICOM_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEFetchDICOM: public IOperation {
public:
    DBOpBEFetchDICOM();
    virtual ~DBOpBEFetchDICOM();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpBEFetchDICOM>(new DBOpBEFetchDICOM());
    }
};

MED_IMG_END_NAMESPACE

#endif