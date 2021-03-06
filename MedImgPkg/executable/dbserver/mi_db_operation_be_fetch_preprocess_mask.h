#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_PREPROCESS_MASK_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_FETCH_PREPROCESS_MASK_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEFetchPreprocessMask: public IOperation {
public:
    DBOpBEFetchPreprocessMask();
    virtual ~DBOpBEFetchPreprocessMask();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpBEFetchPreprocessMask)
};

MED_IMG_END_NAMESPACE

#endif