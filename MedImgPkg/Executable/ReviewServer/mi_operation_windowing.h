#ifndef MED_IMG_OPERATION_WINDOWING_H
#define MED_IMG_OPERATION_WINDOWING_H

#include "MedImgAppCommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE

class OpWindowing : public IOperation {
public:
  OpWindowing();
  virtual ~OpWindowing();

  virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif