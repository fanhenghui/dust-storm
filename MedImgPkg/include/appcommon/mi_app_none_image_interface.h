#ifndef MED_IMG_APPCOMMON_MI_NONE_IMAGE_INTERFACE_H
#define MED_IMG_APPCOMMON_MI_NONE_IMAGE_INTERFACE_H

#include "appcommon/mi_app_common_export.h"
#include "renderalgo/mi_scene_base.h"

MED_IMG_BEGIN_NAMESPACE
class SceneBase;
class AppCommon_Export IAppNoneImage {
public:
    IAppNoneImage() {};
    virtual ~IAppNoneImage() {};

    virtual bool check_dirty() = 0;
    virtual void update() = 0;
    virtual char* serialize_dirty(int& buffer_size) const = 0;
};

MED_IMG_END_NAMESPACE
#endif