#ifndef MED_IMAGING_OBSERVER_H_
#define MED_IMAGING_OBSERVER_H_

#include "MedImgCommon/mi_common_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export IObserver
{
public:
    IObserver(){}
    virtual ~IObserver(){}
    virtual void Update() = 0;
protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif