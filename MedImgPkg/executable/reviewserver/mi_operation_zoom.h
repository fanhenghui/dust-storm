#ifndef MED_IMG_OPERATION_ZOOM_H
#define MED_IMG_OPERATION_ZOOM_H

#include "appcommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE 

class OpZoom : public IOperation {
public:
  OpZoom();
  virtual ~OpZoom();

  virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif