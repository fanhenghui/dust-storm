#include "mi_vr_scene.h"

#include "util/mi_configuration.h"

#include "arithmetic/mi_arithmetic_utils.h"

#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_utils.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_vr_entry_exit_points.h"

#include "mi_volume_infos.h"
#include "mi_ray_caster_canvas.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

struct VRScene::RayEnd
{
    unsigned char *array;//rgb8
    OrthoCamera camera;
    int width;
    int height;
    bool lut_dirty;

    RayEnd():width(0),height(0),array(nullptr),lut_dirty(true) {}
    ~RayEnd() {
        if (array) {
            delete [] array;
            array = nullptr;
        }
    }
    void reset(int w, int h) {
        width = w;
        height = h;
        if (array) {
            delete [] array;
            array = nullptr;
        }
        array = new unsigned char[width*height*3];
    }
};

VRScene::VRScene() : RayCastScene(), _cache_ray_end(new RayEnd()) {
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

VRScene::VRScene(int width, int height) : RayCastScene(width, height), _cache_ray_end(new RayEnd()) {
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

void VRScene::set_color_opacity(std::shared_ptr<ColorTransFunc> color, 
    std::shared_ptr<OpacityTransFunc> opacity, unsigned char label) {
    RayCastScene::set_color_opacity(color, opacity, label);
    _cache_ray_end->lut_dirty = true;
}

void VRScene::cache_ray_end() {
    GLTexture2DPtr ray_end_tex = _canvas->get_color_attach_texture(1);
    if (!ray_end_tex) {
        MI_RENDERALGO_LOG(MI_ERROR) << "ray end texture is null.";
        RENDERALGO_THROW_EXCEPTION("ray end texture is null.");
    }

    bool cache_diety = false;
    if (nullptr == _cache_ray_end->array) {
        _cache_ray_end->reset(_width, _height);
        _cache_ray_end->camera = *_ray_cast_camera;
        _cache_ray_end->lut_dirty = false;
        cache_diety = true;
    } else if (_cache_ray_end->width != _width || _cache_ray_end->height != _height) {
        _cache_ray_end->reset(_width, _height);
        _cache_ray_end->camera = *_ray_cast_camera;
        _cache_ray_end->lut_dirty = false;
        cache_diety = true;
    } else if (_cache_ray_end->camera != *_ray_cast_camera) {
        _cache_ray_end->camera = *_ray_cast_camera;
        _cache_ray_end->lut_dirty = false;
        cache_diety = true;
    } else if (_cache_ray_end->lut_dirty) {
        _cache_ray_end->lut_dirty = false;
        cache_diety = true;
    }

    if (cache_diety) {
        GLUtils::set_pixel_pack_alignment(1);
        GLUtils::set_pixel_unpack_alignment(1);
        ray_end_tex->bind();
        ray_end_tex->download(GL_RGB, GL_UNSIGNED_BYTE, _cache_ray_end->array);
        ray_end_tex->unbind();
    }
}

bool VRScene::get_ray_end(const Point2& pt_cross, Point3& pt_ray_end_world) {
    if (nullptr == _cache_ray_end->array) {
        MI_RENDERALGO_LOG(MI_ERROR) << "cache ray end array is null.";
        RENDERALGO_THROW_EXCEPTION("cache ray end array is null.");
    }

    const int x = int(pt_cross.x);
    const int y = _height - 1 - int(pt_cross.y);//saved texture is flip z
    if (x < 0 || x > _width - 1 || y < 0 || y > _height - 1) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input: " << x << " " << y << " pill when get ray end.";
        return false;
    }

    const unsigned int idx = y*_cache_ray_end->width + x;
    const unsigned char ray_end[3] = {_cache_ray_end->array[idx*3], _cache_ray_end->array[idx*3+1], _cache_ray_end->array[idx*3+2]};

    //MI_RENDERALGO_LOG(MI_DEBUG) << "ray end: " << (int)ray_end[0] << " " << (int)ray_end[1] << " " << (int)ray_end[2];
    if (ray_end[0] == 0 && ray_end[1] == 0 && ray_end[1] == 0) {
        MI_RENDERALGO_LOG(MI_ERROR) << "ray end is all 0.";
        return false;
    }

    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    std::shared_ptr<ImageData> image_data = _volume_infos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(image_data);
    const double dims[3] = {(double)image_data->_dim[0], (double)image_data->_dim[1], (double)image_data->_dim[2]};
    const Point3 pt_v(ray_end[0]/255.0*dims[0], ray_end[1]/255.0*dims[1], ray_end[2]/255.0*dims[2]);
    
    RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
    pt_ray_end_world = _camera_calculator->get_volume_to_world_matrix().transform(pt_v);

    return true;
}

MED_IMG_END_NAMESPACE
