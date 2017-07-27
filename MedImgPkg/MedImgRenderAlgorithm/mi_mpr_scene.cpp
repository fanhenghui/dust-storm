#include "mi_mpr_scene.h"

#include "MedImgUtil/mi_configuration.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"

#include "mi_volume_infos.h"

MED_IMG_BEGIN_NAMESPACE

MPRScene::MPRScene():RayCastScene()
{
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points(new MPREntryExitPoints());
    _entry_exit_points = mpr_entry_exit_points;
    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _entry_exit_points->set_strategy(CPU_BASE);
    }
    else
    {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

MPRScene::MPRScene(int width , int height):RayCastScene(width , height)
{
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points(new MPREntryExitPoints());
    _entry_exit_points = mpr_entry_exit_points;
    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _entry_exit_points->set_strategy(CPU_BASE);
    }
    else
    {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

MPRScene::~MPRScene()
{

}

void MPRScene::place_mpr(ScanSliceType eType)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
    //Calculate MPR placement camera
    _camera_calculator->init_mpr_placement(_ray_cast_camera , eType);
    //Set initial camera to interactor
    _camera_interactor->set_initial_status(_ray_cast_camera);
    //resize because initial camera's ratio between width and height  is 1, but current ratio may not.
    _camera_interactor->resize(_width , _height);

    set_dirty(true);
}

void MPRScene::rotate(const Point2& pre_pt , const Point2& cur_pt)
{
    _camera_interactor->rotate(pre_pt , cur_pt , _width , _height );
    set_dirty(true);
}

void MPRScene::zoom(const Point2& pre_pt , const Point2& cur_pt)
{
    _camera_interactor->zoom(pre_pt , cur_pt , _width , _height );
    set_dirty(true);
}

void MPRScene::pan(const Point2& pre_pt , const Point2& cur_pt)
{
    _camera_interactor->pan(pre_pt , cur_pt , _width , _height );
    set_dirty(true);
}

bool MPRScene::get_volume_position(const Point2& pt_dc , Point3& pos_v)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_data);

    Point2 pt = ArithmeticUtils::dc_to_ndc(pt_dc , _width , _height);

    Matrix4 mat_mvp = _ray_cast_camera->get_view_projection_matrix()*_camera_calculator->get_volume_to_world_matrix();
    mat_mvp.inverse();

    Point3 pos_v_temp = mat_mvp.transform(Point3(pt.x , pt.y , 0.0));
    if (ArithmeticUtils::check_in_bound(pos_v_temp , Point3(volume_data->_dim[0] - 1.0 , volume_data->_dim[1] - 1 , volume_data->_dim[2] - 1)))
    {
        pos_v = pos_v_temp;
        return true;
    }
    else
    {
        return false;
    }
}

bool MPRScene::get_world_position(const Point2& pt_dc , Point3& pos_w)
{
    Point3 pos_v;
    if (get_volume_position(pt_dc , pos_v))
    {
        pos_w = _camera_calculator->get_volume_to_world_matrix().transform(pos_v);
        return true;
    }
    else
    {
        return false;
    }
}

void MPRScene::page(int step)
{
    //TODO should consider oblique MPR
    _camera_calculator->page_orthognal_mpr(_ray_cast_camera , step);
    set_dirty(true);
}

void MPRScene::page_to(int page)
{
    _camera_calculator->page_orthognal_mpr_to(_ray_cast_camera , page);
    set_dirty(true);
}

Plane MPRScene::to_plane() const
{
    Point3 eye = _ray_cast_camera->get_eye();
    Point3 look_at = _ray_cast_camera->get_look_at();

    Vector3 norm = look_at - eye;
    norm.normalize();

    Plane p;
    p._norm = norm;
    p._distance = norm.dot_product(look_at - Point3::S_ZERO_POINT);

    return p;
}

bool MPRScene::get_patient_position(const Point2& pt_dc, Point3& pos_p)
{
    Point3 pt_w;
    if (get_world_position(pt_dc , pt_w))
    {
        pos_p = _camera_calculator->get_world_to_patient_matrix().transform(pt_w);
        return true;
    }
    else
    {
        return false;
    }
}

void MPRScene::set_mask_overlay_mode(MaskOverlayMode mode)
{
    _ray_caster->set_mask_overlay_mode(mode);
}

void MPRScene::set_mask_overlay_color(std::map<unsigned char , RGBAUnit> colors)
{
    _ray_caster->set_mask_overlay_color(colors);
}

void MPRScene::set_mask_overlay_color(RGBAUnit color , unsigned char label)
{
    _ray_caster->set_mask_overlay_color(color , label);
}

Point3 MPRScene::get_entry_point(int x , int y)
{
    Vector4f* entry_array = _entry_exit_points->get_entry_points_array();
    if (entry_array)
    {
        Vector4f v = entry_array[(_height -1 - y)*_width + x];
        return Point3(v._m[0] , v._m[1] , v._m[2]);
    }
    else
    {
        return Point3::S_ZERO_POINT;
    }
}

MED_IMG_END_NAMESPACE
