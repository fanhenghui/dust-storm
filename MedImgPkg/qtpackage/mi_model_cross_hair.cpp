#include "mi_model_cross_hair.h"
#include "arithmetic/mi_intersection_test.h"
#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_arithmetic_utils.h"

#include "io/mi_image_data.h"
#include "renderalgo/mi_volume_infos.h"

#include "renderalgo/mi_mpr_scene.h"

MED_IMG_BEGIN_NAMESPACE

CrosshairModel::CrosshairModel():_is_visible(true)
{
    _pages[0] = 1;
    _pages[1] = 1;
    _pages[2] = 1;
}

CrosshairModel::~CrosshairModel()
{

}

void CrosshairModel::set_mpr_scene(const ScanSliceType (&scan_type)[3] ,const MPRScenePtr (&scenes)[3] ,const RGBUnit (colors)[3])
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(scenes[0]);
    _camera_calculator = scenes[0]->get_camera_calculator();

    for (int i = 0; i<3 ; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(scenes[i]);
        _mpr_scenes[i] = scenes[i];
        _mpr_colors[i] = colors[i];
        _pages[i] =  _camera_calculator->get_default_page(scan_type[i]);
    }

    _location_discrete_w = _camera_calculator->get_default_mpr_center_world();
    _location_contineous_w = _location_discrete_w;
}

void CrosshairModel::get_cross_line(const MPRScenePtr& target_mpr_scene, Line2D (&lines)[2] , RGBUnit (&color)[2])
{
    //1 Choose crossed MPR
    QTWIDGETS_CHECK_NULL_EXCEPTION(target_mpr_scene);
    MPRScenePtr cross_scenes[2] = {nullptr , nullptr};
    int id = 0;
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);
        if (_mpr_scenes[i] == target_mpr_scene)
        {
            continue;
        }
        cross_scenes[id] = _mpr_scenes[i];
        color[id++] = _mpr_colors[i];
    }
    assert(id == 2);


    //2 MPR plane intersected to a plane
    const Matrix4 mat_vp = target_mpr_scene->get_camera()->get_view_projection_matrix();
    Plane plane_target = target_mpr_scene->to_plane();
    for (int i = 0; i<2; ++i)
    {
        Plane p = cross_scenes[i]->to_plane();
        Line3D line_intersect;
        if( IntersectionTest::plane_to_plane(p, plane_target,line_intersect))
        {
            //Project intersected line to screen
            Point3 pt_screen = mat_vp.transform(line_intersect._pt);
            lines[i]._pt = Point2(pt_screen.x , pt_screen.y);
            Vector3 vDir = mat_vp.get_inverse().get_transpose().transform(line_intersect._dir);
            lines[i]._dir = Vector2(vDir.x , vDir.y).get_normalize();
        }
        else
        {
            lines[i]._pt = Point2::S_ZERO_POINT;
            lines[i]._dir = Vector2(0,0);
        }
    }
}

RGBUnit CrosshairModel::get_border_color(MPRScenePtr target_mpr_scene)
{
    for (int i = 0 ; i< 3 ; ++i)
    {
        if (_mpr_scenes[i] == target_mpr_scene)
        {
            return _mpr_colors[i];
        }
    }
    return RGBUnit();
}

bool CrosshairModel::page_to(const std::shared_ptr<MPRScene>& target_mpr_scene, int page)
{
    //1 page target MPR
    int iCurrentPage= get_page(target_mpr_scene);
    if (iCurrentPage == page)
    {
        return false;
    }

    std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(target_mpr_scene->get_camera());
    if( !_camera_calculator->page_orthognal_mpr_to(camera , page))
    {
        return false;
    }

    target_mpr_scene->set_dirty(true);
    set_page_i(target_mpr_scene , page);

    //2 Change cross location
    const Point3 sphere_center = target_mpr_scene->get_camera()->get_look_at();
    const Vector3 vDir = target_mpr_scene->get_camera()->get_view_direction();
    const double dDistance = vDir.dot_product(sphere_center - _location_contineous_w);
    _location_contineous_w += dDistance*vDir;
    _location_discrete_w += dDistance*vDir;

    set_changed();

    return true;
}

bool CrosshairModel::page(const std::shared_ptr<MPRScene>& target_mpr_scene , int step)
{
    //1 page target MPR
    std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(target_mpr_scene->get_camera());
    if( !_camera_calculator->page_orthognal_mpr(camera , step))
    {
        return false;
    }

    target_mpr_scene->set_dirty(true);
    set_page_i(target_mpr_scene , _camera_calculator->get_orthognal_mpr_page(camera));

    //2 Change cross location
    const Point3 sphere_center = target_mpr_scene->get_camera()->get_look_at();
    const Vector3 vDir = target_mpr_scene->get_camera()->get_view_direction();
    const double dDistance = vDir.dot_product(sphere_center - _location_contineous_w);
    _location_contineous_w += dDistance*vDir;
    _location_discrete_w += dDistance*vDir;

    set_changed();

    return true;
}

bool CrosshairModel::locate(const std::shared_ptr<MPRScene>& target_mpr_scene , const Point2& pt_dc )
{
    //1 Get latest location
    Point3 ptV;
    if (!target_mpr_scene->get_volume_position(pt_dc , ptV))
    {
        return false;
    }

    const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
    _location_contineous_w = mat_v2w.transform(ptV);
    _location_discrete_w = mat_v2w.transform(Point3( (double)( (int)ptV.x) , (double)( (int)ptV.y) ,(double)( (int)ptV.z) ));

    //2 Choose crossed MPR
    QTWIDGETS_CHECK_NULL_EXCEPTION(target_mpr_scene);
    MPRScenePtr cross_scenes[2] = {nullptr , nullptr};
    int id = 0;
    int aIdx[2] = {0,0};
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);
        if (_mpr_scenes[i] == target_mpr_scene)
        {
            continue;
        }
        cross_scenes[id] = _mpr_scenes[i];
        aIdx[id++]  = i;
    }
    assert(id == 2);

    //3 Translate crossed MPR( update LookAt and update Page)
    for (int i = 0; i<2 ; ++i)
    {
        std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(cross_scenes[i]->get_camera());
        _camera_calculator->translate_mpr_to(camera, _location_contineous_w);

        cross_scenes[i]->set_dirty(true);
        int page = _camera_calculator->get_orthognal_mpr_page(camera);
        _pages[aIdx[i]] = page;
    }

    set_changed();

    return true;

}

bool CrosshairModel::locate(const Point3& center_w , bool ignore_pan /*= true*/)
{
    //MPR plane paging to the input point slice towards to each normal
    //don't focus the center
    if (!set_center_i(center_w))
    {
        return false;
    }

    for (int i = 0 ; i<3 ; ++ i)
    {
        std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(_mpr_scenes[i]->get_camera());
        _camera_calculator->translate_mpr_to(camera, _location_contineous_w);

        _mpr_scenes[i]->set_dirty(true);
        int page = _camera_calculator->get_orthognal_mpr_page(camera);
        _pages[i] = page;
    }

    //pan center to screen
    if (!ignore_pan)
    {
        for (int i = 0 ; i<3; ++i)
        {
            std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(_mpr_scenes[i]->get_camera());
            const Matrix4 mat_p = camera->get_projection_matrix();
            if (mat_p.has_inverse())
            {
                const Matrix4 mat_vp = camera->get_view_projection_matrix();
                Point3 center_s = mat_vp.transform(center_w);
                const Point3 at_screen = mat_vp.transform(camera->get_look_at());
                const Point3 pt = mat_p.get_inverse().transform(Point3(-center_s.x , -center_s.y , at_screen.z));
                camera->pan(Vector2(pt.x , pt.y));
            }
        }
    }


    set_changed();

    return true;
}

//bool CrosshairModel::locate_focus(const Point3& center_w)
//{
//    //Place MPR center to this center
//
//    return true;
//}

void CrosshairModel::set_page_i(const std::shared_ptr<MPRScene>& target_mpr_scene , int page)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);
        if (_mpr_scenes[i] == target_mpr_scene)
        {
            _pages[i] = page;
            return;
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Cant find certain MPR scene!");
}

int CrosshairModel::get_page(const std::shared_ptr<MPRScene>& target_mpr_scene)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);
        if (_mpr_scenes[i] == target_mpr_scene)
        {
            return _pages[i];
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Cant find certain MPR scene!");
}

bool CrosshairModel::set_center_i(const Point3& center_w)
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(_mpr_scenes[0]);
    std::shared_ptr<VolumeInfos> volume_infos = _mpr_scenes[0]->get_volume_infos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);
    std::shared_ptr<ImageData> volume_data = volume_infos->get_volume();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_data);
    unsigned int *uiDim = volume_data->_dim;

    Point3 ptV = _camera_calculator->get_world_to_volume_matrix().transform(center_w);
    if (!ArithmeticUtils::check_in_bound(ptV , Point3(uiDim[0] , uiDim[1] , uiDim[2])))
    {
        return false;
    }

    _location_contineous_w = center_w;
    
    _location_discrete_w = _camera_calculator->get_volume_to_world_matrix().transform(
        Point3(double((int)ptV.x) , double((int)ptV.y) , double((int)ptV.z) ));

    return true;
}

void CrosshairModel::set_visibility(bool flag)
{
    _is_visible = flag;
}

bool CrosshairModel::get_visibility() const
{
    return _is_visible;
}





MED_IMG_END_NAMESPACE