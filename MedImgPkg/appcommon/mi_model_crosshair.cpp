#include "mi_model_crosshair.h"

#include "arithmetic/mi_intersection_test.h"
#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_arithmetic_utils.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mpr_scene.h"

#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

ModelCrosshair::ModelCrosshair(): _orthogonal(true),_visible(true) {
    _pages[0] = 1;
    _pages[1] = 1;
    _pages[2] = 1;
}

ModelCrosshair::~ModelCrosshair() {

}

void ModelCrosshair::set_mpr_scene(const ScanSliceType(&scan_type)[3],
                                   const MPRScenePtr(&scenes)[3], const RGBUnit(colors)[3]) {
    APPCOMMON_CHECK_NULL_EXCEPTION(scenes[0]);
    _camera_calculator = scenes[0]->get_camera_calculator();

    for (int i = 0; i < 3 ; ++i) {
        APPCOMMON_CHECK_NULL_EXCEPTION(scenes[i]);
        _mpr_scenes[i] = scenes[i];
        _mpr_slice_type[i] = scan_type[i];
        _mpr_colors[i] = colors[i];
        _pages[i] =  _camera_calculator->get_default_page(scan_type[i]);
    }

    _location_discrete_w = _camera_calculator->get_default_mpr_center_world();
    _location_contineous_w = _location_discrete_w;
}

void ModelCrosshair::get_cross_line(const MPRScenePtr& target_mpr_scene, Line2D(&lines_ndc)[2], Point2& cross_ndc, 
                                    Line2D(&lines_dc)[2], Point2& cross_dc, RGBUnit(&color)[2]) {
    //1 get crossed MPR
    APPCOMMON_CHECK_NULL_EXCEPTION(target_mpr_scene);
    MPRScenePtr cross_scenes[2] = {nullptr, nullptr};
    int id = 0;
    for (int i = 0 ; i < 3; ++i) {
        APPCOMMON_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);

        if (_mpr_scenes[i] == target_mpr_scene) {
            continue;
        }

        if (id > 1) {
            MI_APPCOMMON_LOG(MI_ERROR) << "input target mpr scene is not in mpr groups.";
            APPCOMMON_THROW_EXCEPTION("input target mpr scene is not in mpr groups.");
            continue;
        }
        cross_scenes[id] = _mpr_scenes[i];
        color[id] = _mpr_colors[i];
        ++id;
    }

    //2 MPR plane intersected to a plane
    const Matrix4 mat_vp = target_mpr_scene->get_camera()->get_view_projection_matrix();
    Plane plane_target = target_mpr_scene->to_plane();

    for (int i = 0; i < 2; ++i) {
        Plane p = cross_scenes[i]->to_plane();
        Line3D line_intersect;

        if (IntersectionTest::plane_to_plane(p, plane_target, line_intersect)) {
            //project intersected line to screen
            Point3 pt_screen_ndc = mat_vp.transform(_location_contineous_w);
            cross_ndc = Point2(pt_screen_ndc.x, pt_screen_ndc.y);
            lines_ndc[i]._pt = cross_ndc;
            Vector3 vDir = mat_vp.get_inverse().get_transpose().transform(line_intersect._dir);
            lines_ndc[i]._dir = Vector2(vDir.x, vDir.y).get_normalize();

            int width(0),height(0);
            target_mpr_scene->get_display_size(width, height);
            int spill_tag = 0;
            Point2 pt_screen_dc = ArithmeticUtils::ndc_to_dc(cross_ndc, width, height, spill_tag);
            //TODO check spill_tag
            cross_dc = pt_screen_dc;
            lines_dc[i]._pt = pt_screen_dc;
            lines_dc[i]._dir = Vector2(lines_ndc[i]._dir.x, -lines_ndc[i]._dir.y);    
        } else {
            lines_ndc[i]._pt = Point2::S_ZERO_POINT;
            lines_ndc[i]._dir = Vector2(0, 0);
            lines_dc[i]._pt = Point2::S_ZERO_POINT;
            lines_dc[i]._dir = Vector2(0, 0);
            MI_APPCOMMON_LOG(MI_ERROR) << "get cross line failed.";
        }
    }
}

bool ModelCrosshair::get_cross(const MPRScenePtr& target_mpr_scene, Point2& pt_dc) {
    APPCOMMON_CHECK_NULL_EXCEPTION(target_mpr_scene);

    std::shared_ptr<CameraBase> camera = target_mpr_scene->get_camera();
    const Matrix4 matvp = camera->get_view_projection_matrix();
    const Point3 pt_ndc = matvp.transform(_location_discrete_w);
    int width(0), height(0);
    target_mpr_scene->get_display_size(width, height);
    int spill_tag(0);
    pt_dc = ArithmeticUtils::ndc_to_dc(Point2(pt_ndc.x, pt_ndc.y), width, height, spill_tag);
    return spill_tag == 0;
}

RGBUnit ModelCrosshair::get_border_color(MPRScenePtr target_mpr_scene) {
    for (int i = 0 ; i < 3 ; ++i) {
        if (_mpr_scenes[i] == target_mpr_scene) {
            return _mpr_colors[i];
        }
    }
    MI_APPCOMMON_LOG(MI_ERROR) << "input target mpr scene is not in mpr groups.";
    return RGBUnit();
}

bool ModelCrosshair::page_to(const std::shared_ptr<MPRScene>& target_mpr_scene, int page) {
    if (!_orthogonal) {
        MI_APPCOMMON_LOG(MI_WARNING) << "page oblique mpr groups. skip it.";
        return false;
    }

    //page target MPR
    int cur_page = get_page(target_mpr_scene);
    if (cur_page == page) {
        return false;
    }

    std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(target_mpr_scene->get_camera());
    if (!_camera_calculator->page_orthogonal_mpr_to(camera, page)) {
        return false;
    }

    target_mpr_scene->set_dirty(true);
    set_page(target_mpr_scene, page);

    //change cross location
    const Point3 look_center = target_mpr_scene->get_camera()->get_look_at();
    const Vector3 view_dir = target_mpr_scene->get_camera()->get_view_direction();
    const double translate = view_dir.dot_product(look_center - _location_contineous_w);
    _location_contineous_w += translate * view_dir;
    _location_discrete_w += translate * view_dir;

    set_changed();
    return true;
}

bool ModelCrosshair::page(const std::shared_ptr<MPRScene>& target_mpr_scene, int step) {
    //1 page target MPR
    std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(target_mpr_scene->get_camera());
    int cur_page = 0;
    if (!_camera_calculator->page_orthogonal_mpr(camera, step, cur_page)) {
        return false;
    }

    target_mpr_scene->set_dirty(true);
    set_page(target_mpr_scene, cur_page);

    //2 change cross location
    const Point3 look_center = target_mpr_scene->get_camera()->get_look_at();
    const Vector3 view_dir = target_mpr_scene->get_camera()->get_view_direction();
    const double translate = view_dir.dot_product(look_center - _location_contineous_w);
    _location_contineous_w += translate * view_dir;
    _location_discrete_w += translate * view_dir;

    set_changed();
    return true;
}

bool ModelCrosshair::locate(const std::shared_ptr<MPRScene>& target_mpr_scene, const Point2& pt_dc) {
    //1 get latest location
    Point3 pt_volume;
    if (!target_mpr_scene->get_volume_position(pt_dc, pt_volume)) {
        return false;
    }

    const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
    _location_contineous_w = mat_v2w.transform(pt_volume);
    //TODO 需要一个离散和连续的中心点的转换函数
    _location_discrete_w = mat_v2w.transform(Point3((double)((int)pt_volume.x), 
    (double)((int)pt_volume.y), (double)((int)pt_volume.z)));

    //2 Choose crossed MPR
    APPCOMMON_CHECK_NULL_EXCEPTION(target_mpr_scene);
    MPRScenePtr cross_scenes[2] = {nullptr, nullptr};
    int id = 0;
    int cross_mpr_ids[2] = {0, 0};

    for (int i = 0 ; i < 3; ++i) {
        APPCOMMON_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);

        if (_mpr_scenes[i] == target_mpr_scene) {
            continue;
        }

        if (id > 1) {
            MI_APPCOMMON_LOG(MI_ERROR) << "input target mpr scene is not in mpr groups.";
            APPCOMMON_THROW_EXCEPTION("input target mpr scene is not in mpr groups.");
            continue;
        }
        cross_scenes[id] = _mpr_scenes[i];
        cross_mpr_ids[id++]  = i;
    }    

    //3 translate crossed MPR( update LookAt and update Page)
    for (int i = 0; i < 2 ; ++i) {
        std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(cross_scenes[i]->get_camera());
        _camera_calculator->translate_mpr_to(camera, _location_contineous_w);

        cross_scenes[i]->set_dirty(true);
        int page = _camera_calculator->get_orthogonal_mpr_page(camera);
        _pages[cross_mpr_ids[i]] = page;
    }

    set_changed();
    return true;
}

bool ModelCrosshair::locate(const Point3& center_w, bool ignore_pan /*= true*/) {
    //MPR plane paging to the input point slice towards to each normal don't focus the center
    if (!set_center(center_w)) {
        return false;
    }

    for (int i = 0 ; i < 3 ; ++ i) {
        std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>(_mpr_scenes[i]->get_camera());
        _camera_calculator->translate_mpr_to(camera, _location_contineous_w);

        _mpr_scenes[i]->set_dirty(true);
        int page = _camera_calculator->get_orthogonal_mpr_page(camera);
        _pages[i] = page;
    }

    //pan center to screen
    if (!ignore_pan) {
        for (int i = 0 ; i < 3; ++i) {
            std::shared_ptr<OrthoCamera> camera = std::dynamic_pointer_cast<OrthoCamera>
                                                  (_mpr_scenes[i]->get_camera());
            const Matrix4 mat_p = camera->get_projection_matrix();

            if (mat_p.has_inverse()) {
                const Matrix4 mat_vp = camera->get_view_projection_matrix();
                Point3 center_s = 
                mat_vp.transform(center_w);
                const Point3 at_screen = mat_vp.transform(camera->get_look_at());
                const Point3 pt = mat_p.get_inverse().transform(Point3(-center_s.x, -center_s.y, at_screen.z));
                camera->pan(Vector2(pt.x, pt.y));
            }
        }
    }
    set_changed();
    return true;
}

void ModelCrosshair::set_page(const std::shared_ptr<MPRScene>& target_mpr_scene, int page) {
    for (int i = 0 ; i < 3; ++i) {
        APPCOMMON_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);
        if (_mpr_scenes[i] == target_mpr_scene) {
            _pages[i] = page;
            return;
        }
    }

    APPCOMMON_THROW_EXCEPTION("can't find certain MPR scene.");
}

int ModelCrosshair::get_page(const std::shared_ptr<MPRScene>& target_mpr_scene) {
    for (int i = 0 ; i < 3; ++i) {
        APPCOMMON_CHECK_NULL_EXCEPTION(_mpr_scenes[i]);

        if (_mpr_scenes[i] == target_mpr_scene) {
            return _pages[i];
        }
    }

    MI_APPCOMMON_LOG(MI_ERROR) << "can't find certain MPR scene.";
    APPCOMMON_THROW_EXCEPTION("can't find certain MPR scene.");
}

bool ModelCrosshair::set_center(const Point3& center_w) {
    APPCOMMON_CHECK_NULL_EXCEPTION(_mpr_scenes[0]);
    std::shared_ptr<VolumeInfos> volume_infos = _mpr_scenes[0]->get_volume_infos();
    APPCOMMON_CHECK_NULL_EXCEPTION(volume_infos);
    std::shared_ptr<ImageData> volume_data = volume_infos->get_volume();
    APPCOMMON_CHECK_NULL_EXCEPTION(volume_data);
    unsigned int* dim = volume_data->_dim;

    Point3 pt_volume = _camera_calculator->get_world_to_volume_matrix().transform(center_w);
    if (!ArithmeticUtils::check_in_bound(pt_volume, Point3(dim[0], dim[1], dim[2]))) {
        return false;
    }

    _location_contineous_w = center_w;
    _location_discrete_w = _camera_calculator->get_volume_to_world_matrix().transform(
        Point3(double((int)pt_volume.x), double((int)pt_volume.y), double((int)pt_volume.z)));

    return true;
}

void ModelCrosshair::set_visibility(bool flag) {
    _visible = flag;
}

bool ModelCrosshair::get_visibility() const {
    return _visible;
}

void ModelCrosshair::set_mpr_group_orthogonality(bool flag) {
    _orthogonal = flag;
}

bool ModelCrosshair::get_mpr_group_orthogonality() const {
    return _orthogonal;
}

void ModelCrosshair::reset() {
    //TODO
}

Point3 ModelCrosshair::get_cross_location_discrete_world() const {
    return _location_discrete_w;
}

Point3 ModelCrosshair::get_cross_location_contineous_world() const {
    return _location_contineous_w;
}



MED_IMG_END_NAMESPACE