#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_REQUEST_END_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_REQUEST_END_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBERequestEnd: public IOperation {
public:
    DBOpBERequestEnd();
    virtual ~DBOpBERequestEnd();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpBERequestEnd)
};

MED_IMG_END_NAMESPACE

#endif