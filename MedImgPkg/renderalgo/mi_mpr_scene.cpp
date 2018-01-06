#include "mi_mpr_scene.h"

#include "io/mi_configure.h"

#include "arithmetic/mi_arithmetic_utils.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_mpr_entry_exit_points.h"
#include "renderalgo/mi_ray_caster.h"

#include "mi_volume_infos.h"

MED_IMG_BEGIN_NAMESPACE

MPRScene::MPRScene(RayCastingStrategy strategy, GPUPlatform platfrom) : RayCastScene(strategy, platfrom) {
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points(new MPREntryExitPoints(strategy, platfrom));
    _entry_exit_points = mpr_entry_exit_points;
    _name = "MPR Scene";
}

MPRScene::MPRScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom) : 
    RayCastScene(width, height, strategy, platfrom) {
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points(new MPREntryExitPoints(strategy, platfrom));
    _entry_exit_points = mpr_entry_exit_points;
    _name = "MPR Scene";
}

MPRScene::~MPRScene() {}

void MPRScene::initialize() {
    if (GL_BASE == _gpu_platform) {
        RayCastScene::initialize();
    } else {
        //TODO CUDA
        //rewrite scene-base initialize(without scene FBO)
    }
}

void MPRScene::render_to_back() {
    if (GL_BASE == _gpu_platform) {
        SceneBase::render_to_back();
    } else {
        //TODO CUDA
        //memcpy rc-canvas result to host, and draw pixel to BACK
    }
}

void MPRScene::place_mpr(ScanSliceType eType) {
    RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
    // Calculate MPR placement camera
    _camera_calculator->init_mpr_placement(_ray_cast_camera, eType);
    // Set initial camera to interactor
    _camera_interactor->set_initial_status(_ray_cast_camera);
    // resize because initial camera's ratio between width and height  is 1, but
    // current ratio may not.
    _camera_interactor->resize(_width, _height);

    set_dirty(true);
}

void MPRScene::rotate(const Point2& pre_pt, const Point2& cur_pt) {
    _camera_interactor->rotate(pre_pt, cur_pt, _width, _height);
    set_dirty(true);
}

void MPRScene::zoom(const Point2& pre_pt, const Point2& cur_pt) {
    _camera_interactor->zoom(pre_pt, cur_pt, _width, _height);
    set_dirty(true);
}

void MPRScene::pan(const Point2& pre_pt, const Point2& cur_pt) {
    _camera_interactor->pan(pre_pt, cur_pt, _width, _height);
    set_dirty(true);
}

bool MPRScene::get_volume_position(const Point2& pt_dc, Point3& pos_v) {
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_data);

    Point2 pt = ArithmeticUtils::dc_to_ndc(pt_dc, _width, _height);

    Matrix4 mat_mvp = _ray_cast_camera->get_view_projection_matrix() *
                      _camera_calculator->get_volume_to_world_matrix();
    mat_mvp.inverse();

    Point3 pos_v_temp = mat_mvp.transform(Point3(pt.x, pt.y, 0.0));

    if (ArithmeticUtils::check_in_bound(pos_v_temp,
        Point3(volume_data->_dim[0] - 1.0, volume_data->_dim[1] - 1, volume_data->_dim[2] - 1))) {
        pos_v = pos_v_temp;
        return true;
    } else {
        return false;
    }
}

bool MPRScene::get_world_position(const Point2& pt_dc, Point3& pos_w) {
    Point3 pos_v;

    if (get_volume_position(pt_dc, pos_v)) {
        pos_w = _camera_calculator->get_volume_to_world_matrix().transform(pos_v);
        return true;
    } else {
        return false;
    }
}

void MPRScene::page(int step) {
    // TODO should consider oblique MPR
    int cur_page = 0;
    _camera_calculator->page_orthogonal_mpr(_ray_cast_camera, step , cur_page);
    set_dirty(true);
}

void MPRScene::page_to(int page) {
    _camera_calculator->page_orthogonal_mpr_to(_ray_cast_camera, page);
    set_dirty(true);
}

Plane MPRScene::to_plane() const {
    Point3 eye = _ray_cast_camera->get_eye();
    Point3 look_at = _ray_cast_camera->get_look_at();

    Vector3 norm = look_at - eye;
    norm.normalize();

    Plane p;
    p._norm = norm;
    p._distance = norm.dot_product(look_at - Point3::S_ZERO_POINT);

    return p;
}

bool MPRScene::get_patient_position(const Point2& pt_dc, Point3& pos_p) {
    Point3 pt_w;

    if (get_world_position(pt_dc, pt_w)) {
        pos_p = _camera_calculator->get_world_to_patient_matrix().transform(pt_w);
        return true;
    } else {
        return false;
    }
}

void MPRScene::set_mask_overlay_mode(MaskOverlayMode mode) {
    _ray_caster->set_mask_overlay_mode(mode);
}

void MPRScene::set_mask_overlay_color(
    std::map<unsigned char, RGBAUnit> colors) {
    _ray_caster->set_mask_overlay_color(colors);
}

void MPRScene::set_mask_overlay_color(RGBAUnit color, unsigned char label) {
    _ray_caster->set_mask_overlay_color(color, label);
}

void MPRScene::set_mask_overlay_opacity(float opacity) {
    _ray_caster->set_mask_overlay_opacity(opacity);
}

MED_IMG_END_NAMESPACE
