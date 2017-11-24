#ifndef MED_IMG_APPCOMMON_MI_COMMON_CONTROLLER_INTERFACE_H_
#define MED_IMG_APPCOMMON_MI_COMMON_CONTROLLER_INTERFACE_H_

#include "appcommon/mi_app_common_export.h"
#include <map>
#include <memory>
#include <string>

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