#ifndef MED_IMG_OBSERVER_SCENE_CONTAINER_H
#define MED_IMG_OBSERVER_SCENE_CONTAINER_H

#include "MedImgQtPackage/mi_qt_package_export.h"
#include "MedImgUtil/mi_observer_interface.h"

class SceneContainer;

MED_IMG_BEGIN_NAMESPACE

class QtPackage_Export SceneContainerObserver : public IObserver
{
public: 
    SceneContainerObserver();
    virtual ~SceneContainerObserver();

    virtual void update(int code_id = 0);

    void add_scene_container(SceneContainer* container);
    void set_scene_container(const std::vector<SceneContainer*>& containers);

    void clear_scene_container();
protected:
private:
    std::vector<SceneContainer*> _scene_containers; 
};

MED_IMG_END_NAMESPACE

#endif