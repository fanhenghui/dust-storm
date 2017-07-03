#ifndef MED_IMG_OBSERVER_H_
#define MED_IMG_OBSERVER_H_

#include "MedImgUtil/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

class Util_Export IObserver
{
public:
    IObserver(){}
    virtual ~IObserver(){}

    virtual void update(int code_id = 0) = 0;
protected:
private:
};

MED_IMG_END_NAMESPACE

#endif