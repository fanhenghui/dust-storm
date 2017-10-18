#include "mi_vr_scene.h"

#include "util/mi_configuration.h"

#include "arithmetic/mi_arithmetic_utils.h"

#include "glresource/mi_gl_texture_2d.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_vr_entry_exit_points.h"

#include "mi_volume_infos.h"
#include "mi_ray_caster_canvas.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

VRScene::VRScene() : RayCastScene() {
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points(
        new VREntryExitPoints());
    vr_entry_exit_points->set_brick_filter_item(BF_WL);
    _entry_exit_points = vr_entry_exit_points;

    if (CPU == Configuration::instance()->get_processing_unit_type()) {
        _entry_exit_points->set_strategy(CPU_BASE);
    } else {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

VRScene::VRScene(int width, int height) : RayCastScene(width, height) {
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points(
        new VREntryExitPoints());
    vr_entry_exit_points->set_brick_filter_item(BF_WL);
    _entry_exit_points = vr_entry_exit_points;

    if (CPU == Configuration::instance()->get_processing_unit_type()) {
        _entry_exit_points->set_strategy(CPU_BASE);
    } else {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

VRScene::~VRScene() {}

void VRScene::initialize() {
    SceneBase::initialize();     

    _canvas->initialize(true);
    _entry_exit_points->initialize();
}

void VRScene::rotate(const Point2& pre_pt, const Point2& cur_pt) {
    if (pre_pt != cur_pt) {
        _camera_interactor->rotate(pre_pt, cur_pt, _width, _height);
        set_dirty(true);
    }
}

void VRScene::zoom(const Point2& pre_pt, const Point2& cur_pt) {
    if (pre_pt != cur_pt) {
        _camera_interactor->zoom(pre_pt, cur_pt, _width, _height);
        set_dirty(true);
    }
}

void VRScene::pan(const Point2& pre_pt, const Point2& cur_pt) {
    if (pre_pt != cur_pt) {
        _camera_interactor->pan(pre_pt, cur_pt, _width, _height);
        set_dirty(true);
    }
}

void VRScene::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos) {
    RayCastScene::set_volume_infos(volume_infos);

    // place default VR
    _camera_calculator->init_vr_placement(_ray_cast_camera);

    // Set initial camera to interactor
    _camera_interactor->set_initial_status(_ray_cast_camera);

    // resize because initial camera's ratio between width and height  is 1, but
    // current ratio may not.
    _camera_interactor->resize(_width, _height);

    // initialize vr ee
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);

    std::shared_ptr<ImageData> volume = _volume_infos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume);
    AABB default_aabb;
    default_aabb._min = Point3::S_ZERO_POINT;
    default_aabb._max.x = static_cast<double>(volume->_dim[0]);
    default_aabb._max.y = static_cast<double>(volume->_dim[1]);
    default_aabb._max.z = static_cast<double>(volume->_dim[2]);
    vr_entry_exit_points->set_bounding_box(default_aabb);

    vr_entry_exit_points->set_brick_pool(_volume_infos->get_brick_pool());
}

void VRScene::set_window_level(float ww, float wl, unsigned char label) {
    RayCastScene::set_window_level(ww, wl, label);
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_window_level(ww, wl, label, false);
}

void VRScene::set_global_window_level(float ww, float wl) {
    RayCastScene::set_global_window_level(ww, wl);
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_window_level(ww, wl, 0, true);
}

void VRScene::set_visible_labels(std::vector<unsigned char> labels) {
    RayCastScene::set_visible_labels(labels);
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_visible_labels(labels);
}

void VRScene::set_bounding_box(const AABB& aabb) {
    // TODO check aabb
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_bounding_box(aabb);
}

void VRScene::set_clipping_plane(std::vector<Plane> planes) {
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_clipping_plane(planes);
}

void VRScene::set_proxy_geometry(ProxyGeometry pg_type) {
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);
    vr_entry_exit_points->set_proxy_geometry(pg_type);
}

void VRScene::pre_render_i() {
    RayCastScene::pre_render_i();

    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);

    if (_ray_caster->get_composite_mode() == COMPOSITE_DVR) {
        if (_ray_caster->get_mask_mode() == MASK_MULTI_LABEL) {
            vr_entry_exit_points->set_brick_filter_item(BF_MASK | BF_WL);
        } else {
            vr_entry_exit_points->set_brick_filter_item(BF_WL);
        }
    } else {
        vr_entry_exit_points->set_brick_filter_item(BF_WL);
    }
}

bool VRScene::get_ray_end(const Point2& pt_cross, Point3& pt_ray_end_world) {
    GLTexture2DPtr ray_end_tex = _canvas->get_color_attach_texture(1);
    if (!ray_end_tex) {
        MI_RENDERALGO_LOG(MI_ERROR) << "ray end texture is null.";
        return false;
    }
    const int x = int(pt_cross.x);
    const int y = int(pt_cross.y);
    if (x < 0 || x > _width - 1 || y < 0 || y > _height - 1) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input: " << x << " " << y << " pill when get ray end.";
        return false;
    }

    unsigned char pixel_value[3] = {0,0,0};
    {
        if (_canvas->get_color_attach_texture(1)) {
            std::stringstream ss;
            ss << "/home/wangrui22/data/output_para2_" << _width << "_" << _height << ".raw";
            _canvas->debug_output_color1(ss.str());
            _canvas->debug_output_color(ss.str() + "_color");
        }
    }
    ray_end_tex->bind();
    // {
    //     std::unique_ptr<unsigned char[]> color_array(new unsigned char[_width * _height * 3]);
    //     ray_end_tex->download(GL_RGB, GL_UNSIGNED_BYTE,color_array.get());
    //     std::stringstream ss;
    //     ss << "/home/wangrui22/data/output_para1_" << _width << "_" << _height << ".raw";
    //     std::ofstream out(ss.str(), std::ios::out | std::ios::binary);
    //     if (out.is_open()) {
    //         out.write((char*)color_array.get(), _width * _height * 3);
    //     }
    //     out.close();
    // }
    ray_end_tex->read_pixels(GL_RGB, GL_UNSIGNED_BYTE, x, _height - y -1, 1, 1, pixel_value);
    ray_end_tex->unbind();
    MI_RENDERALGO_LOG(MI_DEBUG) << "ray end: " << pixel_value[0] << " " << pixel_value[1] << " " << pixel_value[2];
    if (pixel_value[0] == 0 && pixel_value[1] == 0 && pixel_value[1] == 0) {
        return false;
    }

    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    std::shared_ptr<ImageData> image_data = _volume_infos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(image_data);
    const double dims[3] = {(double)image_data->_dim[0], (double)image_data->_dim[1], (double)image_data->_dim[2]};
    const Point3 pt_v(pixel_value[0]/dims[0], pixel_value[1]/dims[1], pixel_value[2]/dims[2]);
    
    RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
    pt_ray_end_world = _camera_calculator->get_volume_to_world_matrix().transform(pt_v);

    return true;
}

MED_IMG_END_NAMESPACE
