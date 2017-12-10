#ifndef MEDIMG_UTIL_MI_CONTROLLER_INTERFACE_H
#define MEDIMG_UTIL_MI_CONTROLLER_INTERFACE_H

#include "util/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

class IController {
public:
    IController() {}
    virtual ~IController() {}
    virtual void initialize() = 0;
    virtual void finalize() = 0;

private:
    DISALLOW_COPY_AND_ASSIGN(IController);
};

MED_IMG_END_NAMESPACE
#endif