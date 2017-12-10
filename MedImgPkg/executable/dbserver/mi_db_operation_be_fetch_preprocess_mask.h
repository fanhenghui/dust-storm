#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_PREPROCESS_MASK_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_PREPROCESS_MASK_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEFetchPreprocessMask: public IOperation {
public:
    DBOpBEFetchPreprocessMask();
    virtual ~DBOpBEFetchPreprocessMask();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpBEFetchPreprocessMask>(new DBOpBEFetchPreprocessMask());
    }
};

MED_IMG_END_NAMESPACE

#endif