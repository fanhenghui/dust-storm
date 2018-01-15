#include "mi_vr_scene.h"

#include "io/mi_configure.h"

#include "arithmetic/mi_arithmetic_utils.h"

#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_resource_manager_container.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_vr_entry_exit_points.h"

#include "mi_volume_infos.h"
#include "mi_ray_caster_canvas.h"
#include "mi_render_algo_logger.h"
#include "mi_graphic_object_navigator.h"

#include "mi_gpu_image_compressor.h"

MED_IMG_BEGIN_NAMESPACE

struct VRScene::RayEnd
{
    unsigned char *array;//rgb8
    OrthoCamera camera;
    int width;
    int height;
    bool lut_dirty;

    RayEnd():array(nullptr),width(0),height(0),lut_dirty(true) {}
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

class EntryExitPointsResizeCallback : public IEntryExitPointsResizeCallback {
public:
    EntryExitPointsResizeCallback(std::shared_ptr<RayCaster> rc):_ray_caster(rc){}
    virtual ~EntryExitPointsResizeCallback() {}
    virtual void execute(int width, int height) {
        if (_ray_caster) {
            _ray_caster->on_entry_exit_points_resize(width, height);
        }
    }
private:
    std::shared_ptr<RayCaster> _ray_caster;
};

VRScene::VRScene(RayCastingStrategy strategy, GPUPlatform platfrom) : 
    RayCastScene(strategy, platfrom), _cache_ray_end(new RayEnd()) {
    //--------------------------------------------------//
    //VR entry exit just support GL platform
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points( new VREntryExitPoints(strategy, GL_BASE));
    vr_entry_exit_points->set_brick_filter_item(BF_WL);
    _entry_exit_points = vr_entry_exit_points;
    _name = "VR Scene";

    //register entry exit points resize callback
    _entry_exit_points->register_resize_callback(
        std::shared_ptr<IEntryExitPointsResizeCallback>(new EntryExitPointsResizeCallback(_ray_caster)));
}

VRScene::VRScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom) : 
    RayCastScene(width, height, strategy, platfrom), _cache_ray_end(new RayEnd()) {
    //--------------------------------------------------//
    //VR entry exit just support GL platform
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points( new VREntryExitPoints(strategy, GL_BASE));
    vr_entry_exit_points->set_brick_filter_item(BF_WL);
    _entry_exit_points = vr_entry_exit_points;
    _name = "VR Scene";

    //register entry exit points resize callback
    _entry_exit_points->register_resize_callback(
        std::shared_ptr<IEntryExitPointsResizeCallback>(new EntryExitPointsResizeCallback(_ray_caster)));
}

VRScene::~VRScene() {}

void VRScene::initialize() {
    if (GL_BASE == _gpu_platform) {
        if (!_scene_fbo) {//as init flag
            SceneBase::initialize();

            _canvas->initialize(true);
            _entry_exit_points->initialize();
            _navigator->initialize();
        }
    } else {
        //-----------------------------------------------------------//
        //forbid base's initialize, just create FBO with color-attach-0 (For graphic rendering)
        if (!_scene_fbo) {//as init flag
            CHECK_GL_ERROR;

            GLUtils::set_pixel_pack_alignment(1);

            _scene_fbo = GLResourceManagerContainer::instance()
                ->get_fbo_manager()->create_object("scene base FBO");
            _scene_fbo->initialize();
            _scene_fbo->set_target(GL_FRAMEBUFFER);

            _scene_color_attach_0 = GLResourceManagerContainer::instance()
                ->get_texture_2d_manager()->create_object("scene base color attachment 0");
            _scene_color_attach_0->initialize();
            _scene_color_attach_0->bind();
            GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
            GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
            _scene_color_attach_0->load(GL_RGB8, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

            _scene_depth_attach = GLResourceManagerContainer::instance()
                ->get_texture_2d_manager()->create_object("scene base depth attachment");
            _scene_depth_attach->initialize();
            _scene_depth_attach->bind();
            GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
            GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
            _scene_depth_attach->load(GL_DEPTH_COMPONENT16, _width, _height,
                GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);

            //bind texture to FBO
            _scene_fbo->bind();
            _scene_fbo->attach_texture(GL_COLOR_ATTACHMENT0, _scene_color_attach_0);
            _scene_fbo->attach_texture(GL_DEPTH_ATTACHMENT, _scene_depth_attach);
            _scene_fbo->unbind();

            CHECK_GL_ERROR;

            _res_shield.add_shield<GLFBO>(_scene_fbo);
            _res_shield.add_shield<GLTexture2D>(_scene_color_attach_0);
            _res_shield.add_shield<GLTexture2D>(_scene_depth_attach);

            //-----------------------------------------------------------//
            //initialize others
            _canvas->initialize(true);
            _entry_exit_points->initialize();
            _navigator->initialize();

            //-------------------------------//
            // GPU compressor (set rc-canvas's color-attach-0 as input)
            //-------------------------------//
            std::vector<int> qualitys(2);
            qualitys[0] = _compress_ld_quality;
            qualitys[1] = _compress_hd_quality;

            _compressor->set_image(GPUCanvasPairPtr(
                new GPUCanvasPair(_canvas->get_color_attach_texture()->get_cuda_resource())), qualitys);
        }
    }
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

    // initialize vr entry exit points
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

void VRScene::pre_render() {
    RayCastScene::pre_render();

    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points =
        std::dynamic_pointer_cast<VREntryExitPoints>(_entry_exit_points);

    //roll-back to custom's proxy geometry 
    switch (_ray_caster->get_composite_mode())
    {
    case COMPOSITE_DVR: {
        if (_ray_caster->get_mask_mode() == MASK_MULTI_LABEL) {
            // Use all visible labels
            std::map<unsigned char, Vector2f> wls;
            _ray_caster->get_visible_window_levels(wls);
            vr_entry_exit_points->set_window_levels(wls);
            vr_entry_exit_points->set_brick_filter_item(BF_MASK | BF_WL);
        }
        else {
            // Just use label 0
            std::map<unsigned char, Vector2f> wls;
            _ray_caster->get_window_levels(wls);
            auto it_label_0 = wls.find(0);
            if (it_label_0 == wls.end()) {
                RENDERALGO_THROW_EXCEPTION("non-mask dvr should set wl to label 0");
            }
            std::map<unsigned char, Vector2f> wl_0;
            wl_0.insert(std::make_pair(0, it_label_0->second));
            vr_entry_exit_points->set_window_levels(wl_0);
            vr_entry_exit_points->set_brick_filter_item(BF_WL);
        }
        break;
    }
    case COMPOSITE_MIP:
    case COMPOSITE_MINIP:
    case COMPOSITE_AVERAGE: {
        if (_ray_caster->get_mask_mode() == MASK_MULTI_LABEL) {
            // Set all visible labels to global window levels
            float ww, wl;
            _ray_caster->get_global_window_level(ww, wl);
            std::map<unsigned char, Vector2f> wls;
            const std::vector<unsigned char> visible_labels = _ray_caster->get_visible_labels();
            for (auto it = visible_labels.begin(); it != visible_labels.end(); ++it) {
                wls[*it] = Vector2f(ww, wl);
            }
            vr_entry_exit_points->set_window_levels(wls);
            vr_entry_exit_points->set_brick_filter_item(BF_MASK | BF_WL);
        } else {
            // Just use global WL
            float ww, wl;
            _ray_caster->get_global_window_level(ww, wl);
            vr_entry_exit_points->set_global_window_level(ww, wl);
            vr_entry_exit_points->set_brick_filter_item(BF_WL);
        }
        break;
    }
    default:
        RENDERALGO_THROW_EXCEPTION("invalid composite mode.");
        break;
    }
}

void VRScene::set_color_opacity(std::shared_ptr<ColorTransFunc> color, 
    std::shared_ptr<OpacityTransFunc> opacity, unsigned char label) {
    RayCastScene::set_color_opacity(color, opacity, label);
    _cache_ray_end->lut_dirty = true;
}

void VRScene::cache_ray_end() {
    if (GL_BASE == _gpu_platform) {
        GLTexture2DPtr ray_end_tex = _canvas->get_color_attach_texture(1)->get_gl_resource();
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
        }
        else if (_cache_ray_end->width != _width || _cache_ray_end->height != _height) {
            _cache_ray_end->reset(_width, _height);
            _cache_ray_end->camera = *_ray_cast_camera;
            _cache_ray_end->lut_dirty = false;
            cache_diety = true;
        }
        else if (_cache_ray_end->camera != *_ray_cast_camera) {
            _cache_ray_end->camera = *_ray_cast_camera;
            _cache_ray_end->lut_dirty = false;
            cache_diety = true;
        }
        else if (_cache_ray_end->lut_dirty) {
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
