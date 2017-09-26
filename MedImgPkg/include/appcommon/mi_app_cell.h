#ifndef MED_IMH_APP_COMMON_CELL_H_
#define MED_IMH_APP_COMMON_CELL_H_

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

    void set_noneimage(std::shared_ptr<IAppNoneImage> noneimage);
    std::shared_ptr<IAppNoneImage> get_noneimage();

protected:
private:
    std::shared_ptr<SceneBase> _scene;
    std::shared_ptr<IAppNoneImage> _none_img;
};

MED_IMG_END_NAMESPACE

#endif