#ifndef MED_IMG_OPERATION_INIT_H
#define MED_IMG_OPERATION_INIT_H

#include "appcommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE

class OpInit : public IOperation {
public:
  OpInit();
  virtual ~OpInit();

  virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif