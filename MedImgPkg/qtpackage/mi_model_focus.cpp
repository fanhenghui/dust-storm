#include "mi_model_focus.h"

MED_IMG_BEGIN_NAMESPACE 

FocusModel::FocusModel():_container(nullptr)
{

}

FocusModel::~FocusModel()
{

}

void FocusModel::set_focus_scene_container(SceneContainer* container )
{
    _container = container;
}

SceneContainer*  FocusModel::get_focus_scene_container() const
{
    return _container;
}

MED_IMG_END_NAMESPACE