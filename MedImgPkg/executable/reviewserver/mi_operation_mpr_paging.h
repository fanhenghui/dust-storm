#ifndef MED_IMG_OPERATION_MPR_PAGING_H
#define MED_IMG_OPERATION_MPR_PAGING_H

#include "mi_review_common.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE 

class OpMPRPaging : public IOperation
{
public:
    OpMPRPaging();
    virtual ~OpMPRPaging();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif