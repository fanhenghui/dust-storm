#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_PACS_RETRIEVE_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_PACS_RETRIEVE_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEPACSRetrieve: public IOperation {
public:
    DBOpBEPACSRetrieve();
    virtual ~DBOpBEPACSRetrieve();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpBEPACSRetrieve)
};

MED_IMG_END_NAMESPACE

#endif