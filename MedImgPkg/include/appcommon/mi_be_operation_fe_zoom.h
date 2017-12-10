#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ZOOM_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ZOOM_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEZoom : public IOperation {
public:
    BEOpFEZoom();
    virtual ~BEOpFEZoom();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFEZoom>(new BEOpFEZoom());
    }
protected:
private:
};

MED_IMG_END_NAMESPACE

#endif