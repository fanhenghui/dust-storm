#ifndef MED_IMAGING_MODEL_FOCUS_H_
#define MED_IMAGING_MODEL_FOCUS_H_

#include "MedImgCommon/mi_model_interface.h"

class SceneContainer;

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export FocusModel : public IModel
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

MED_IMAGING_END_NAMESPACE
#endif