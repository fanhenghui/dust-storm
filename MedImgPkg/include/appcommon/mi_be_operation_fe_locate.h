#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_LOCATE_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_LOCATE_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"
#include "arithmetic/mi_point2.h"

MED_IMG_BEGIN_NAMESPACE
class MPRScene;
class VRScene;
class AppCell;
class AppCommon_Export BEOpFELocate : public IOperation {
public:
    BEOpFELocate();
    virtual ~BEOpFELocate();

    virtual int execute();

    CREATE_EXTENDS_OP(BEOpFELocate)

private:
    int mpr_locate(std::shared_ptr<AppCell> cell, std::shared_ptr<MPRScene> mpr_scene, const Point2& pt);
    int vr_locate(std::shared_ptr<AppCell> cell, std::shared_ptr<VRScene> vr_scene, const Point2& pt);
};
MED_IMG_END_NAMESPACE

#endif