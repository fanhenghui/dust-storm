#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_PACS_RETRIEVE_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_PACS_RETRIEVE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEPACSRetrieve: public IOperation {
public:
    DBOpBEPACSRetrieve();
    virtual ~DBOpBEPACSRetrieve();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpBEPACSRetrieve>(new DBOpBEPACSRetrieve());
    }
};

MED_IMG_END_NAMESPACE

#endif