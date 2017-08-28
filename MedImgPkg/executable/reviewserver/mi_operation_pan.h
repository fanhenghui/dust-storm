#ifndef MED_IMG_OPERATION_PAN_H
#define MED_IMG_OPERATION_PAN_H

#include "appcommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE 

class OpPan : public IOperation {
public:
  OpPan();
  virtual ~OpPan();

  virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif