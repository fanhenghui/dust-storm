#ifndef MED_IMG_APPCOMMON_MI_OPERATION_LOCATE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_LOCATE_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"
#include "arithmetic/mi_point2.h"

MED_IMG_BEGIN_NAMESPACE
class MPRScene;
class VRScene;
class AppCell;
class AppCommon_Export OpLocate : public IOperation {
public:
    OpLocate();
    virtual ~OpLocate();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpLocate>(new OpLocate());
    }

private:
    int mpr_locate_i(std::shared_ptr<AppCell> cell, std::shared_ptr<MPRScene> mpr_scene, const Point2& pt);
    int vr_locate_i(std::shared_ptr<AppCell> cell, std::shared_ptr<VRScene> vr_scene, const Point2& pt);
};
MED_IMG_END_NAMESPACE

#endif