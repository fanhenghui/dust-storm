#include "mi_ray_cast_scene.h"

#include "util/mi_file_util.h"

#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_arithmetic_utils.h"

#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_texture_1d_array.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"

#include "cudaresource/mi_cuda_gl_texture_2d.h"
#include "cudaresource/mi_cuda_surface_2d.h"
#include "cudaresource/mi_cuda_resource_manager.h"

#include "mi_camera_calculator.h"
#include "mi_camera_interactor.h"
#include "mi_entry_exit_points.h"
#include "mi_ray_caster.h"
#include "mi_ray_caster_canvas.h"
#include "mi_volume_infos.h"
#include "mi_brick_pool.h"
#include "mi_graphic_object_navigator.h"
#include "mi_transfer_function_texture.h"
#include "mi_gpu_image_compressor.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

RayCastScene::RayCastScene(RayCastingStrategy strategy, GPUPlatform platfrom) : SceneBase(platfrom),
    _strategy(strategy), _global_ww(0), _global_wl(0) {
    _ray_cast_camera.reset(new OrthoCamera());
    _camera = _ray_cast_camera;

    _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

    _ray_caster.reset(new RayCaster(strategy, platfrom));

    _canvas.reset(new RayCasterCanvas(strategy, platfrom));

    _transfer_funcion_texture.reset(new TransferFunctionTexture(strategy, platfrom));
    _transfer_funcion_texture->initialize(L_8);

    _navigator_margin[0] = 20;
    _navigator_margin[1] = 20;
    _navigator_window_ratio = 4.5f;
    _navigator_vis = false;
    _navigator.reset(new GraphicObjectNavigator());
    const int min_size = int((std::min)(_width, _height)/_navigator_window_ratio);
    _navigator->set_navi_position(_width - min_size - _navigator_margin[0], _navigator_margin[1] , min_size, min_size);
}

RayCastScene::RayCastScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom)
    : SceneBase(platfrom, width, height), _strategy(strategy), _global_ww(0), _global_wl(0) {
    _ray_cast_camera.reset(new OrthoCamera());
    _camera = _ray_cast_camera;

    _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

    _ray_caster.reset(new RayCaster(strategy, platfrom));

    _canvas.reset(new RayCasterCanvas(strategy, platfrom));
    _canvas->set_display_size(_width, _height);

    _transfer_funcion_texture.reset(new TransferFunctionTexture(strategy, platfrom));
    _transfer_funcion_texture->initialize(L_8);

    _navigator_margin[0] = 20;
    _navigator_margin[1] = 20;
    _navigator_window_ratio = 4.5f;
    _navigator_vis = false;
    _navigator.reset(new GraphicObjectNavigator());
    const int min_size = int((std::min)(_width, _height)/_navigator_window_ratio);
    _navigator->set_navi_position(_width - min_size - _navigator_margin[0], _navigator_margin[1] , min_size, min_size);
}

RayCastScene::~RayCastScene() {}

void RayCastScene::initialize() {
    if (GL_BASE == _gpu_platform) {
        if (!_scene_fbo) {//as init flag
            SceneBase::initialize();

            _canvas->initialize();
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
        }        
    }
}

void RayCastScene::set_display_size(int width, int height) {
    if (width == _width && height == _height) {
        return;
    }

    SceneBase::set_display_size(width, height);
    _canvas->set_display_size(width, height);
    _entry_exit_points->set_display_size(width, height);
    _camera_interactor->resize(width, height);

    const int min_size = int((std::min)(_width, _height)/_navigator_window_ratio);
    _navigator->set_navi_position(_width - min_size - _navigator_margin[0], _navigator_margin[1] , min_size, min_size);

    if (GL_BASE) {
        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_2D, _scene_color_attach_1, GL_CLAMP_TO_EDGE, GL_LINEAR,
            GL_RGB8, _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    }
}

void RayCastScene::render_to_back() {
    if (CUDA_BASE == _gpu_platform) {
        //kernel memcpy ray-casting result back to FBO color-attachment0
        CudaSurface2DPtr canvas_color0 = _canvas->get_color_attach_texture()->get_cuda_resource();
        std::unique_ptr<unsigned char[]> rgba(new unsigned char[_width * _height * 4]);
        std::unique_ptr<unsigned char[]> rgb(new unsigned char[_width * _height * 3]);

        if (0 != canvas_color0->download(_width*_height * 4, rgba.get())) {
            MI_RENDERALGO_LOG(MI_ERROR) << "download cuda rc result failed when render to back.";
            return;
        }
        for (int i = 0; i < _width*_height; ++i) {
            rgb[i * 3] = rgba[i * 4];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
       
        CHECK_GL_ERROR;

        GLUtils::set_pixel_pack_alignment(1);
        GLUtils::set_pixel_unpack_alignment(1);

        _scene_color_attach_0->bind();
        _scene_color_attach_0->update(0, 0, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, rgb.get());
        _scene_color_attach_0->unbind();

        CHECK_GL_ERROR;
    }

    SceneBase::render_to_back();
}

void RayCastScene::pre_render() {
    // refresh volume & mask & their infos
    _volume_infos->refresh();

    // scene FBO , ray casting program ...
    initialize();

    CHECK_GL_ERROR;

    // GL resource update (discard)
    GLResourceManagerContainer::instance()->update_all();

    CHECK_GL_ERROR;

    // GL texture update
    GLTextureCache::instance()->process_cache();

    CHECK_GL_ERROR;

    // scene base pre-render to reset gpu compressor when display size changed
    if (GL_BASE == _gpu_platform) {
        SceneBase::pre_render();
    }
    else if (_gpujpeg_encoder_dirty) {
        std::vector<int> qualitys(2);
        qualitys[0] = _compress_ld_quality;
        qualitys[1] = _compress_hd_quality;
        _compressor->set_image(GPUCanvasPairPtr(new
            GPUCanvasPair(_canvas->get_color_attach_texture()->get_cuda_resource())), qualitys);
        _gpujpeg_encoder_dirty = false;

    }

    CHECK_GL_ERROR;

    _navigator->set_camera(_camera);
}

void RayCastScene::render() {
    pre_render();

    // Skip render scene
    if (!get_dirty()) {
        return;
    }

    CHECK_GL_ERROR;

    //////////////////////////////////////////////////////////////////////////
    // TODO other common graphic object rendering list

    //////////////////////////////////////////////////////////////////////////
    // 1 Ray casting
    _entry_exit_points->calculate_entry_exit_points();
    std::cout << "width: " << _width << ", height: " << _height;
    _entry_exit_points->debug_output_entry_points("d:/temp/entry_points.rgb");
    _ray_caster->render();

    //////////////////////////////////////////////////////////////////////////
    // 2 Mapping ray casting result to Scene FBO
    if (GL_BASE == _gpu_platform) {
        //----------------------------------------------------------------//
        // GL based : 
        // 1. render graphic (scene FBO color-attach-0)
        // 2. ray-casting (scene FBO color-attach-0)
        // 3. render navigator (scene FBO color-attach-0)
        // 4. mapping and flip vertically color-attach-0 to color-attach-1 for next compressing 
        //----------------------------------------------------------------//

        glViewport(0, 0, _width, _height);

        _scene_fbo->bind();
        glDrawBuffer(GL_COLOR_ATTACHMENT0);

        glViewport(0, 0, _width, _height);
        glClearColor(0, 0, 0, 0);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_TEXTURE_2D);
        _canvas->get_color_attach_texture()->get_gl_resource()->bind();
        if (_ray_caster->map_quarter_canvas()) {
            glBegin(GL_QUADS);
            glTexCoord2f(0.0, 0.0);
            glVertex2f(-1.0, -1.0);
            glTexCoord2f(0.5, 0.0);
            glVertex2f(1.0, -1.0);
            glTexCoord2f(0.5, 0.5);
            glVertex2f(1.0, 1.0);
            glTexCoord2f(0.0, 0.5);
            glVertex2f(-1.0, 1.0);
            glEnd();
        }
        else {
            glBegin(GL_QUADS);
            glTexCoord2f(0.0, 0.0);
            glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0, 0.0);
            glVertex2f(1.0, -1.0);
            glTexCoord2f(1.0, 1.0);
            glVertex2f(1.0, 1.0);
            glTexCoord2f(0.0, 1.0);
            glVertex2f(-1.0, 1.0);
            glEnd();
        }
        _canvas->get_color_attach_texture()->get_gl_resource()->unbind();

        //render navigator in the end
        if (_navigator_vis) {
            _navigator->render();
        }

        // CHECK_GL_ERROR;
        // glPopAttrib();//TODO Here will give a GL_INVALID_OPERATION error !!!
        // CHECK_GL_ERROR;

        _scene_fbo->unbind();

        //flip vertically for download
        glBindFramebuffer(GL_READ_FRAMEBUFFER, _scene_fbo->get_id());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _scene_fbo->get_id());
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glDrawBuffer(GL_COLOR_ATTACHMENT1);
        glBlitFramebuffer(0, _height, _width, 0, 0, 0, _width, _height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        CHECK_GL_ERROR;
    } else {
        //-------------------------------------------------------------------------------//
        // CUDA based (skip mapping): 
        // 1. render graphic(scene FBO color-attach-0)
        // 2. do ray-casting with graphic(kernel blend)
        // 3. kernel navigator ray tracing
        // *. render to back should do mapping, off-screen render skip it.
        // *. GPU compressor will use rc-canvas's result(RGBA->fliped RGB) as input.
        //-------------------------------------------------------------------------------//
        
        //TODO CUDA (maybe do nothing)
    }

    set_dirty(false);
}

void RayCastScene::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos) {
    try {
        RENDERALGO_CHECK_NULL_EXCEPTION(volume_infos);
        _volume_infos = volume_infos;

        std::shared_ptr<ImageData> volume = _volume_infos->get_volume();
        RENDERALGO_CHECK_NULL_EXCEPTION(volume);

        std::shared_ptr<ImageData> mask = _volume_infos->get_mask();
        RENDERALGO_CHECK_NULL_EXCEPTION(mask);

        std::shared_ptr<ImageDataHeader> data_header =
            _volume_infos->get_data_header();
        RENDERALGO_CHECK_NULL_EXCEPTION(data_header);

        // Camera calculator
        _camera_calculator = volume_infos->get_camera_calculator();

        // Entry exit points
        _entry_exit_points->set_volume_data(volume);
        _entry_exit_points->set_camera(_camera);
        _entry_exit_points->set_display_size(_width, _height);
        _entry_exit_points->set_camera_calculator(_camera_calculator);

        // Ray caster
        _ray_caster->set_canvas(_canvas);
        _ray_caster->set_entry_exit_points(_entry_exit_points);
        _ray_caster->set_camera(_camera);
        _ray_caster->set_camera_calculator(_camera_calculator);
        _ray_caster->set_volume_data(volume);
        _ray_caster->set_mask_data(mask);

        if (GPU_BASE == _strategy) {
            // set texture
            RENDERALGO_CHECK_NULL_EXCEPTION(_transfer_funcion_texture);
            _ray_caster->set_pseudo_color_texture(_transfer_funcion_texture->get_pseudo_color_texture(), S_TRANSFER_FUNC_WIDTH);
            _ray_caster->set_color_opacity_texture_array(_transfer_funcion_texture->get_color_opacity_texture_array());
            _ray_caster->set_volume_data_texture(volume_infos->get_volume_texture());
            _ray_caster->set_mask_data_texture(volume_infos->get_mask_texture());
        }

        _navigator->set_volume_info(volume_infos);

        set_dirty(true);
    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << "set volume infos failed with exception: " << e.what();
        assert(false);
        throw e;
    }
}

void RayCastScene::set_mask_label_level(LabelLevel label_level) {
    _ray_caster->set_mask_label_level(label_level);
    RENDERALGO_CHECK_NULL_EXCEPTION(_transfer_funcion_texture);
    _transfer_funcion_texture->initialize(label_level);

    set_dirty(true);
}

void RayCastScene::set_sample_step(float sample_step) {
    _ray_caster->set_sample_step(sample_step);
    set_dirty(true);
}

void RayCastScene::set_visible_labels(std::vector<unsigned char> labels) {
    for (auto it = labels.begin(); it != labels.end(); ++it) {
        if (*it == 0) {
            RENDERALGO_THROW_EXCEPTION("visible labels contain zero");
        }
    }

    if (_volume_infos && !labels.empty()) {
        //here can't update empty visible labels(none-mask)
        _volume_infos->get_brick_pool()->add_visible_labels_cache(labels);
    }

    if (_ray_caster->get_visible_labels() != labels) {
        _ray_caster->set_visible_labels(labels);
        set_dirty(true);
    }
}

std::vector<unsigned char> RayCastScene::get_visible_labels() const {
    return _ray_caster->get_visible_labels();
}

void RayCastScene::set_window_level(float ww, float wl, unsigned char label) {
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    auto it = _window_levels.find(label);

    if (it == _window_levels.end()) {
        _window_levels.insert(std::make_pair(label, Vector2f(ww, wl)));
    } else {
        it->second.set_x(ww);
        it->second.set_y(wl);
    }

    _volume_infos->get_volume()->regulate_normalize_wl(ww, wl);
    _ray_caster->set_window_level(ww, wl, label);

    set_dirty(true);
}

int RayCastScene::get_window_level(float& ww, float& wl, unsigned char label) const {
    const std::map<unsigned char, Vector2f>::const_iterator it =
        _window_levels.find(label);

    if (it == _window_levels.end()) {
        return -1;
    } else {
        ww = it->second.get_x();
        wl = it->second.get_y();
        return 0;
    }
}

void RayCastScene::set_global_window_level(float ww, float wl) {
    _global_ww = ww;
    _global_wl = wl;

    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    _volume_infos->get_volume()->regulate_wl(ww, wl);

    _ray_caster->set_global_window_level(ww, wl);

    set_dirty(true);
}

void RayCastScene::set_mask_mode(MaskMode mode) {
    if (_ray_caster->get_mask_mode() != mode) {
        _ray_caster->set_mask_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_composite_mode(CompositeMode mode) {
    if (_ray_caster->get_composite_mode() != mode) {
        _ray_caster->set_composite_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_interpolation_mode(InterpolationMode mode) {
    if (_ray_caster->get_interpolation_mode() != mode) {
        _ray_caster->set_interpolation_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_shading_mode(ShadingMode mode) {
    if (_ray_caster->get_shading_mode() != mode) {
        _ray_caster->set_shading_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_color_inverse_mode(ColorInverseMode mode) {
    if (_ray_caster->get_color_inverse_mode() != mode) {
        _ray_caster->set_color_inverse_mode(mode);
        set_dirty(true);
    }
}

MaskMode RayCastScene::get_mask_mode() const {
    return _ray_caster->get_mask_mode();
}

CompositeMode RayCastScene::get_composite_mode() const {
    return _ray_caster->get_composite_mode();
}

InterpolationMode RayCastScene::get_interpolation_mode() const {
    return _ray_caster->get_interpolation_mode();
}

ShadingMode RayCastScene::get_shading_mode() const {
    return _ray_caster->get_shading_mode();
}

ColorInverseMode RayCastScene::get_color_inverse_mode() const {
    return _ray_caster->get_color_inverse_mode();
}

void RayCastScene::set_ambient_color(float r, float g, float b, float factor) {
    _ray_caster->set_ambient_color(r, g, b, factor);
    set_dirty(true);
}

void RayCastScene::set_material(const Material& m, unsigned char label) {
    _ray_caster->set_material(m, label);
    set_dirty(true);
}

void RayCastScene::set_pseudo_color(std::shared_ptr<ColorTransFunc> color) {
    RENDERALGO_CHECK_NULL_EXCEPTION(_transfer_funcion_texture);
    _transfer_funcion_texture->set_pseudo_color(color);
    set_dirty(true);
}

void RayCastScene::set_color_opacity(std::shared_ptr<ColorTransFunc> color,
                                     std::shared_ptr<OpacityTransFunc> opacity,
                                     unsigned char label) {

    RENDERALGO_CHECK_NULL_EXCEPTION(_transfer_funcion_texture);
    _transfer_funcion_texture->set_color_opacity(color, opacity, label);
    set_dirty(true);
}

void RayCastScene::set_test_code(int test_code) {
    _ray_caster->set_test_code(test_code);

    set_dirty(true);
}

void RayCastScene::get_global_window_level(float& ww, float& wl) const {
    ww = _global_ww;
    wl = _global_wl;
}

std::shared_ptr<VolumeInfos> RayCastScene::get_volume_infos() const {
    return _volume_infos;
}

std::shared_ptr<CameraCalculator> RayCastScene::get_camera_calculator() const {
    return _camera_calculator;
}

Point2 RayCastScene::project_point_to_screen(const Point3& pt) const {
    const Matrix4 mat_vp = _camera->get_view_projection_matrix();
    Point3 pt_ndc = mat_vp.transform(pt);
    return ArithmeticUtils::ndc_to_dc(Point2(pt_ndc.x, pt_ndc.y), _width, _height);
}

void RayCastScene::set_downsample(bool flag) {
    SceneBase::set_downsample(flag);
    _ray_caster->set_downsample(flag);
}

void RayCastScene::set_expected_fps(int fps) {
    _ray_caster->set_expected_fps(fps);
}

int RayCastScene::get_expected_fps() const {
    return _ray_caster->get_expected_fps();
}

void RayCastScene::set_navigator_visibility(bool flag) {
    _navigator_vis = flag;
}

void RayCastScene::set_navigator_para(int x_margin, int y_margin, float ratio) {
    _navigator_margin[0] = x_margin;
    _navigator_margin[1] = y_margin;
    _navigator_window_ratio = ratio;
}

MED_IMG_END_NAMESPACE