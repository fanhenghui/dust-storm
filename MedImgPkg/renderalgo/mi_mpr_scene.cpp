#include "mi_mpr_scene.h"

#include "arithmetic/mi_arithmetic_utils.h"

#include "io/mi_image_data.h"

#include "glresource/mi_gl_utils.h"
#include "cudaresource/mi_cuda_surface_2d.h"

#include "mi_camera_calculator.h"
#include "mi_camera_interactor.h"
#include "mi_mpr_entry_exit_points.h"
#include "mi_ray_caster.h"
#include "mi_ray_caster_canvas.h"
#include "mi_volume_infos.h"
#include "mi_graphic_object_navigator.h"
#include "mi_gpu_image_compressor.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

MPRScene::MPRScene(RayCastingStrategy strategy, GPUPlatform platfrom) : RayCastScene(strategy, platfrom), _mpr_init(false){
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points(new MPREntryExitPoints(strategy, platfrom));
    _entry_exit_points = mpr_entry_exit_points;
    _name = "MPR Scene";
}

MPRScene::MPRScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom) : 
    RayCastScene(width, height, strategy, platfrom), _mpr_init(false) {
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points(new MPREntryExitPoints(strategy, platfrom));
    _entry_exit_points = mpr_entry_exit_points;
    _name = "MPR Scene";
}

MPRScene::~MPRScene() {}

void MPRScene::initialize() {
    if (GL_BASE == _gpu_platform) {
        RayCastScene::initialize();
    } else if (!_mpr_init) {
        //-----------------------------------------------------------//
        // CUDA based MPR needn't scene FBO(without graphic)

        //-----------------------------------------------------------//
        // initialize others
        _canvas->initialize();
        _entry_exit_points->initialize();
        _navigator->initialize();

        //-------------------------------//
        // GPU compressor (set rc-canvas's color-attach-0 as input)
        //-------------------------------//
        std::vector<int> qualitys(2);
        qualitys[0] = _compress_ld_quality;
        qualitys[1] = _compress_hd_quality;

        _compressor->set_image(GPUCanvasPairPtr(new
            GPUCanvasPair(_canvas->get_color_attach_texture()->get_cuda_resource())), qualitys);

        _mpr_init = true;
    }
}

void MPRScene::render_to_back() {
    if (GL_BASE == _gpu_platform) {
        SceneBase::render_to_back();
    } else {
        //-------------------------------------------------------------//
        // memcpy rc-canvas result to host, and draw pixel to BACK
        // just for test
        CudaSurface2DPtr canvas_color0 = _canvas->get_color_attach_texture()->get_cuda_resource();
        std::unique_ptr<unsigned char[]> rgba(new unsigned char[_width * _height * 4]);

        if (0 != canvas_color0->download(_width*_height * 4, rgba.get())) {
            MI_RENDERALGO_LOG(MI_ERROR) << "download cuda rc result failed when render to back.";
            return;
        }

        CHECK_GL_ERROR;
        glDrawPixels(_width, _height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.get());
        CHECK_GL_ERROR;
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
    if (_camera_calculator->page_orthogonal_mpr_to(_ray_cast_camera, page)) {
        set_dirty(true);
    }
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
