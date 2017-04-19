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
    virtual void Update();
    void AddSceneContainer(SceneContainer* pContainer);
    void SetSceneContainer(const std::vector<SceneContainer*>& vecConainers);
    void ClearSceneContainer();
protected:
private:
    std::vector<SceneContainer*> m_vecContainers; 
};

MED_IMAGING_END_NAMESPACE

#endif