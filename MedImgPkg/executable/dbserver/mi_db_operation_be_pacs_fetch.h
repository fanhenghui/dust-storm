#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEPACSFetch: public IOperation {
public:
    DBOpBEPACSFetch();
    virtual ~DBOpBEPACSFetch();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpBEPACSFetch>(new DBOpBEPACSFetch());
    }
};

MED_IMG_END_NAMESPACE

#endif