#ifndef MED_IMG_MI_DB_OPERATION_PACS_RETRIEVE_H
#define MED_IMG_MI_DB_OPERATION_PACS_RETRIEVE_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpPACSRetrieve: public IOperation {
public:
    DBOpPACSRetrieve();
    virtual ~DBOpPACSRetrieve();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpPACSRetrieve>(new DBOpPACSRetrieve());
    }
};

MED_IMG_END_NAMESPACE

#endif