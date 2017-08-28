#ifndef MED_IMG_MODEL_FOCUS_H_
#define MED_IMG_MODEL_FOCUS_H_

#include "MedImgQtPackage/mi_qt_package_export.h"
#include "util/mi_model_interface.h"

class SceneContainer;

MED_IMG_BEGIN_NAMESPACE

class QtPackage_Export FocusModel : public IModel
{
public:
    FocusModel();
    virtual ~FocusModel();

    void set_focus_scene_container(SceneContainer* container);
    SceneContainer*  get_focus_scene_container() const;

protected:
private:
    SceneContainer* _container;
};

MED_IMG_END_NAMESPACE
#endif