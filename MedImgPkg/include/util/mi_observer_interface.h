#ifndef MEDIMGUTIL_MI_OBSERVER_INTERFACE_H
#define MEDIMGUTIL_MI_OBSERVER_INTERFACE_H

#include "util/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

class Util_Export IObserver {
public:
    IObserver() {}
    virtual ~IObserver() {}

    virtual void update(int code_id = 0) = 0;
protected:
private:
};

MED_IMG_END_NAMESPACE

#endif