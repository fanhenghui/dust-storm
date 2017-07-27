#include "mi_vr_scene.h"

#include "MedImgUtil/mi_configuration.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_vr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"

#include "mi_volume_infos.h"


MED_IMG_BEGIN_NAMESPACE

VRScene::VRScene():RayCastScene()
{
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points(new VREntryExitPoints());
    _entry_exit_points = vr_entry_exit_points;
    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _entry_exit_points->set_strategy(CPU_BASE);
    }
    else
    {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

VRScene::VRScene(int width , int height)
{
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points(new VREntryExitPoints());
    _entry_exit_points = vr_entry_exit_points;
    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _entry_exit_points->set_strategy(CPU_BASE);
    }
    else
    {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

VRScene::~VRScene()
{

}

void VRScene::rotate(const Point2& pre_pt , const Point2& cur_pt)
{

}

void VRScene::zoom(const Point2& pre_pt , const Point2& cur_pt)
{

}

void VRScene::pan(const Point2& pre_pt , const Point2& cur_pt)
{

}

void VRScene::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos)
{
    RayCastScene::set_volume_infos(volume_infos);

    //place default VR
    _camera_calculator->init_vr_placement(_ray_cast_camera);

    //initialize bounding box

}

void VRScene::set_bounding_box(const AABB& aabb)
{
    //TODO check aabb
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points = 
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_bounding_box(aabb);
}

void VRScene::set_clipping_plane(std::vector<Plane> planes)
{
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points = 
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_clipping_plane(planes);
}

void VRScene::set_proxy_geometry(ProxyGeometry pg_type)
{
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points = 
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_proxy_geometry(pg_type);
}

MED_IMG_END_NAMESPACE

