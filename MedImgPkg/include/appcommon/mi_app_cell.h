#ifndef MED_IMG_APPCOMMON_MI_APP_CELL_H
#define MED_IMG_APPCOMMON_MI_APP_CELL_H

#include "appcommon/mi_app_common_export.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class SceneBase;
class IAppNoneImage;
class AppCommon_Export AppCell {
public:
    AppCell();
    virtual ~AppCell();

    void set_scene(std::shared_ptr<SceneBase> scene);
    std::shared_ptr<SceneBase> get_scene();

    void set_none_image(std::shared_ptr<IAppNoneImage> noneimage);
    std::shared_ptr<IAppNoneImage> get_none_image();

protected:
private:
    std::shared_ptr<SceneBase> _scene;
    std::shared_ptr<IAppNoneImage> _none_img;
};

MED_IMG_END_NAMESPACE

#endif