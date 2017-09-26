#include "mi_app_cell.h"
#include "renderalgo/mi_scene_base.h"
#include "mi_app_none_image_interface.h"

MED_IMG_BEGIN_NAMESPACE

AppCell::AppCell() {

}

AppCell::~AppCell() {

}

void AppCell::set_scene(std::shared_ptr<SceneBase> scene) {
    _scene = scene;
}

std::shared_ptr<SceneBase> AppCell::get_scene() {
    return _scene;
}

void AppCell::set_noneimage(std::shared_ptr<IAppNoneImage> noneimage) {
    _none_img = noneimage;
}

std::shared_ptr<IAppNoneImage> AppCell::get_noneimage() {
    return _none_img;
}

MED_IMG_END_NAMESPACE