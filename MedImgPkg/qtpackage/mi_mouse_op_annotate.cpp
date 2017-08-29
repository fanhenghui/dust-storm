#include "mi_mouse_op_annotate.h"

#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_vector3.h"

#include "renderalgo/mi_scene_base.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mask_label_store.h"

#include "mi_model_voi.h"


MED_IMG_BEGIN_NAMESPACE

const static std::string  S_DEFAULT_NODULE_TYPE = std::string("W");

MouseOpAnnotate::MouseOpAnnotate():_is_pin(false),_diameter(0.0),_current_label(0)
{

}

MouseOpAnnotate::~MouseOpAnnotate()
{

}

void MouseOpAnnotate::press(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    _pre_point = pt;
    _is_pin = false;
    _diameter = 0.0;

    //New voi
    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    if (scene&&_model)
    {
        Point3 sphere_center;
        if(scene->get_patient_position(Point2(pt.x() , pt.y()) , sphere_center))
        {
            //Get VOI center
            _is_pin = true;
            _center = sphere_center;
            _diameter = 0.0;
            _current_label = MaskLabelStore::instance()->acquire_label();
            _model->add_voi(medical_imaging::VOISphere(_center , _diameter , S_DEFAULT_NODULE_TYPE) , _current_label);
            _model->notify(VOIModel::ADD_VOI);
        }
    }
}

void MouseOpAnnotate::move(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    if (_is_pin&&_model)
    {
        Point3 pt_mpr;
        std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
        if(scene->get_patient_position(Point2(pt.x() , pt.y()) , pt_mpr))
        {
            //Get VOI center
            Vector3 v = pt_mpr - _center;
            _diameter = v.magnitude()*2.0;

            _model->modify_voi_list_rear(medical_imaging::VOISphere(_center , _diameter));
            _model->notify(VOIModel::MODIFYING);
        }
    }

    _pre_point = pt;
}

void MouseOpAnnotate::release(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    if (!_model->get_vois().empty())
    {
        if( (--_model->get_vois().end())->diameter < 0.1f )
        {
            MaskLabelStore::instance()->recycle_label(_current_label);
            _current_label = 0;
            _model->remove_voi_list_rear();
            _model->notify(VOIModel::DELETE_VOI);
        }
        else
        {
            _model->set_changed();
            _model->notify(VOIModel::MODIFY_COMPLETED);
        }

    }
    _pre_point = pt;
}

void MouseOpAnnotate::double_click(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpAnnotate::set_voi_model(std::shared_ptr<VOIModel> model)
{
    _model = model;
}

void MouseOpAnnotate::wheel_slide(int )
{

}

MED_IMG_END_NAMESPACE