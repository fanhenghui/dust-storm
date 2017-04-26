#include "mi_mouse_op_annotate.h"

#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_vector3.h"

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

const std::string NODULE_TYPE_GGN = std::string("GGN");
const std::string NODULE_TYPE_AAH = std::string("AAH");

MouseOpAnnotate::MouseOpAnnotate():_is_pin(false),_diameter(0.0)
{

}

MouseOpAnnotate::~MouseOpAnnotate()
{

}

void MouseOpAnnotate::press(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    _pre_point = pt;
    _is_pin = false;
    _diameter = 0.0;

    //New voi
    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    if (scene&&_model)
    {
        Point3 sphere_center;
        if(scene->get_patient_position(Point2(pt.x() , pt.y()) , sphere_center))
        {
            //Get VOI center
            _is_pin = true;
            _center = sphere_center;
            _diameter = 0.0;
            _model->add_voi_sphere(medical_imaging::VOISphere(_center , _diameter , NODULE_TYPE_GGN));
            _model->notify();
        }
    }
}

void MouseOpAnnotate::move(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    if (_is_pin&&_model)
    {
        Point3 pt_mpr;
        std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(_scene);
        if(scene->get_patient_position(Point2(pt.x() , pt.y()) , pt_mpr))
        {
            //Get VOI center
            Vector3 v = pt_mpr - _center;
            _diameter = v.magnitude()*2.0;

            _model->modify_voi_sphere_list_rear(medical_imaging::VOISphere(_center , _diameter));
            _model->notify();
        }
    }

    _pre_point = pt;
}

void MouseOpAnnotate::release(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    if (!_model->get_voi_spheres().empty())
    {
        if( (--_model->get_voi_spheres().end())->diameter < 0.1f )
        {
            _model->remove_voi_sphere_list_rear();
            _model->notify();
        }
    }
    _pre_point = pt;
}

void MouseOpAnnotate::double_click(const QPointF& pt)
{
    if (!_scene)
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

MED_IMAGING_END_NAMESPACE