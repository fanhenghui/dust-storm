#ifndef MED_IMG_MI_DB_OPERATION_QUERY_PROCESS_MASK_H
#define MED_IMG_MI_DB_OPERATION_QUERY_PROCESS_MASK_H

#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpQueryPreprocessMask: public IOperation {
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