#ifndef MED_IMG_MI_DB_OPERATION_PACS_FETCH_H
#define MED_IMG_MI_DB_OPERATION_PACS_FETCH_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpPACSFetch: public IOperation {
public:
    DBOpPACSFetch();
    virtual ~DBOpPACSFetch();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpPACSFetch>(new DBOpPACSFetch());
    }
};

MED_IMG_END_NAMESPACE

#endif