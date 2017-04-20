#include "mi_observer_scene_container.h"
#include "mi_scene_container.h"

MED_IMAGING_BEGIN_NAMESPACE

SceneContainerObserver::SceneContainerObserver()
{

}

SceneContainerObserver::~SceneContainerObserver()
{

}

void SceneContainerObserver::update()
{
    for (auto it = m_vecContainers.begin() ; it != m_vecContainers.end() ; ++it)
    {
        if( (*it))
        {
            (*it)->update();
        }
    }
}

void SceneContainerObserver::add_scene_container(SceneContainer* pContainer)
{
    m_vecContainers.push_back(pContainer);
}

void SceneContainerObserver::set_scene_container(const std::vector<SceneContainer*>& vecConainers)
{
    m_vecContainers = vecConainers;
}

void SceneContainerObserver::clear_scene_container()
{
    m_vecContainers.clear();
}

MED_IMAGING_END_NAMESPACE