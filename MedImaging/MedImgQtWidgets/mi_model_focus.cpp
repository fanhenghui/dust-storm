#include "mi_model_focus.h"

MED_IMAGING_BEGIN_NAMESPACE

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

MED_IMAGING_END_NAMESPACE