#ifndef MED_IMG_OPERATION_BE_READY_H
#define MED_IMG_OPERATION_BE_READY_H

#include "MedImgAppCommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE

class OpBEReady : public IOperation {
public:
  OpBEReady();
  virtual ~OpBEReady();

  virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif