#include "mi_app_cell.h"
#include "renderalgo/mi_scene_base.h"

MED_IMG_BEGIN_NAMESPACE

AppCell::AppCell()
{

}

AppCell::~AppCell()
{

}

void AppCell::set_scene(std::shared_ptr<SceneBase> scene)
{
    _scene = scene;
}

std::shared_ptr<SceneBase> AppCell::get_scene()
{
    return _scene;
}

MED_IMG_END_NAMESPACE