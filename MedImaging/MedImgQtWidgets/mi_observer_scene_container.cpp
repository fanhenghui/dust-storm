#include "mi_observer_scene_container.h"
#include "mi_scene_container.h"

MED_IMAGING_BEGIN_NAMESPACE

SceneContainerObserver::SceneContainerObserver()
{

}

SceneContainerObserver::~SceneContainerObserver()
{

}

void SceneContainerObserver::update(int code_id)
{
    for (auto it = _scene_containers.begin() ; it != _scene_containers.end() ; ++it)
    {
        if( (*it))
        {
            (*it)->update_scene();
        }
    }
}

void SceneContainerObserver::add_scene_container(SceneContainer* container)
{
    _scene_containers.push_back(container);
}

void SceneContainerObserver::set_scene_container(const std::vector<SceneContainer*>& containers)
{
    _scene_containers = containers;
}

void SceneContainerObserver::clear_scene_container()
{
    _scene_containers.clear();
}

MED_IMAGING_END_NAMESPACE