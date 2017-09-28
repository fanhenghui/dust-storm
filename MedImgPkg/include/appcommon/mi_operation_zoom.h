#ifndef MED_IMG_APPCOMMON_MI_OPERATION_ZOOM_H
#define MED_IMG_APPCOMMON_MI_OPERATION_ZOOM_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpZoom : public IOperation {
public:
    OpZoom();
    virtual ~OpZoom();

    virtual int execute();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif