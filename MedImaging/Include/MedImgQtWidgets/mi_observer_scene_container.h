#ifndef MED_IMAGING_OBSERVER_SCENE_CONTAINER_H
#define MED_IMAGING_OBSERVER_SCENE_CONTAINER_H

#include "MedImgCommon/mi_observer_interface.h"

class SceneContainer;

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export SceneContainerObserver : public IObserver
{
public: 
    SceneContainerObserver();
    virtual ~SceneContainerObserver();
    virtual void update();
    void add_scene_container(SceneContainer* pContainer);
    void set_scene_container(const std::vector<SceneContainer*>& vecConainers);
    void clear_scene_container();
protected:
private:
    std::vector<SceneContainer*> m_vecContainers; 
};

MED_IMAGING_END_NAMESPACE

#endif