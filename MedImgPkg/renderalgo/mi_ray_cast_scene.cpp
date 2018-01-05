#include "mi_ray_cast_scene.h"

#include "io/mi_configure.h"
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

#include "mi_camera_calculator.h"
#include "mi_camera_interactor.h"
#include "mi_color_transfer_function.h"
#include "mi_entry_exit_points.h"
#include "mi_opacity_transfer_function.h"
#include "mi_ray_caster.h"
#include "mi_ray_caster_canvas.h"
#include "mi_volume_infos.h"
#include "mi_brick_pool.h"
#include "mi_graphic_object_navigator.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

RayCastScene::RayCastScene() : SceneBase(), _global_ww(0), _global_wl(0) {
    _ray_cast_camera.reset(new OrthoCamera());
    _camera = _ray_cast_camera;

    _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

    _ray_caster.reset(new RayCaster());

    _canvas.reset(new RayCasterCanvas());

    if (CPU == Configure::instance()->get_processing_unit_type()) {
        _ray_caster->set_strategy(CPU_BASE);
    } else {
        _ray_caster->set_strategy(GPU_BASE);
    }

    init_default_color_texture();

    _navigator_margin[0] = 20;
    _navigator_margin[1] = 20;
    _navigator_window_ratio = 4.5f;
    _navigator_vis = false;
    _navigator.reset(new GraphicObjectNavigator());
    const int min_size = int((std::min)(_width, _height)/_navigator_window_ratio);
    _navigator->set_navi_position(_width - min_size - _navigator_margin[0], _navigator_margin[1] , min_size, min_size);
}

RayCastScene::RayCastScene(int width, int height)
    : SceneBase(width, height), _global_ww(0), _global_wl(0) {
    _ray_cast_camera.reset(new OrthoCamera());
    _camera = _ray_cast_camera;

    _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

    _ray_caster.reset(new RayCaster());

    _canvas.reset(new RayCasterCanvas());
    _canvas->set_display_size(_width, _height);

    if (CPU == Configure::instance()->get_processing_unit_type()) {
        _ray_caster->set_strategy(CPU_BASE);
    } else {
        _ray_caster->set_strategy(GPU_BASE);
    }

    init_default_color_texture();

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
    SceneBase::initialize();     

    _canvas->initialize();
    _entry_exit_points->initialize();
    _navigator->initialize();
}

void RayCastScene::set_display_size(int width, int height) {
    SceneBase::set_display_size(width, height);
    _canvas->set_display_size(width, height);
    _entry_exit_points->set_display_size(width, height);
    _camera_interactor->resize(width, height);

    const int min_size = int((std::min)(_width, _height)/_navigator_window_ratio);
    _navigator->set_navi_position(_width - min_size - _navigator_margin[0], _navigator_margin[1] , min_size, min_size);
}

void RayCastScene::pre_render() {
    // refresh volume & mask & their infos
    _volume_infos->refresh();

    // scene FBO , ray casting program ...
    initialize();

    // entry exit points initialize
    _entry_exit_points->initialize();

    // GL resource update (discard)
    GLResourceManagerContainer::instance()->update_all();

    // GL texture update
    GLTextureCache::instance()->process_cache();

    // scene base pre-render to recreate jpeg encoder(this must be call after gl
    // resource update)
    SceneBase::pre_render();

    _navigator->set_camera(_camera);
}

void RayCastScene::init_default_color_texture() {
    if (GPU == Configure::instance()->get_processing_unit_type()) {
        // initialize gray pseudo color texture
        if (!_pseudo_color_texture) {
            _pseudo_color_texture = GLResourceManagerContainer::instance()->
                get_texture_1d_manager()->create_object("pseudo color");
            _res_shield.add_shield<GLTexture1D>(_pseudo_color_texture);

            unsigned char* gray_array = new unsigned char[S_TRANSFER_FUNC_WIDTH * 3];

            for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
                gray_array[i * 3] = static_cast<unsigned char>(255.0f * (float)i / (float)S_TRANSFER_FUNC_WIDTH);
                gray_array[i * 3 + 1] = gray_array[i * 3];
                gray_array[i * 3 + 2] = gray_array[i * 3];
            }

            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_1D, _pseudo_color_texture, GL_CLAMP_TO_EDGE, GL_LINEAR,
                GL_RGB8, S_TRANSFER_FUNC_WIDTH, 0, 0, GL_RGB, GL_UNSIGNED_BYTE, (char*)gray_array);
        }

        if (!_color_opacity_texture_array) {
            _color_opacity_texture_array = GLResourceManagerContainer::instance()->
                get_texture_1d_array_manager()->create_object("color opacity texture array");

            const int tex_num = 8; // default mask level
            unsigned char* rgba = new unsigned char[S_TRANSFER_FUNC_WIDTH * tex_num * 4];
            memset(rgba, 0, S_TRANSFER_FUNC_WIDTH * tex_num * 4);

            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array, GL_CLAMP_TO_EDGE,
                GL_LINEAR, GL_RGBA8, S_TRANSFER_FUNC_WIDTH, tex_num, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, (char*)rgba);
        }
    } else {
        // TODO gray pseudo array
    }
}

void RayCastScene::render() {
    pre_render();

    // Skip render scene
    if (!get_dirty()) {
        return;
    }

    MI_RENDERALGO_LOG(MI_TRACE) << "IN ray cast scene render.";

    CHECK_GL_ERROR;

    //////////////////////////////////////////////////////////////////////////
    // TODO other common graphic object rendering list

    //////////////////////////////////////////////////////////////////////////
    // 1 Ray casting
    // glPushAttrib(GL_ALL_ATTRIB_BITS);

    // glViewport(0, 0, _width, _height);
    // glClearColor(0, 0, 0, 0);
    // glClearDepth(1.0);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _entry_exit_points->calculate_entry_exit_points();

    _ray_caster->render();
    // glPopAttrib();

    //////////////////////////////////////////////////////////////////////////
    // 2 Mapping ray casting result to Scene FBO (<><>flip vertically<><>)
    // glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0, 0, _width, _height);

    _scene_fbo->bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT1);
    
    glViewport(0, 0, _width, _height);
    glClearColor(0, 0, 0, 0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    _canvas->get_color_attach_texture()->bind();
    if (_ray_caster->map_quarter_canvas()) {
        // glBegin(GL_QUADS);
        // glTexCoord2f(0.0, 0.5);
        // glVertex2f(-1.0, -1.0);
        // glTexCoord2f(0.5, 0.5);
        // glVertex2f(1.0, -1.0);
        // glTexCoord2f(0.5, 0.0);
        // glVertex2f(1.0, 1.0);
        // glTexCoord2f(0.0, 0.0);
        // glVertex2f(-1.0, 1.0);
        // glEnd();

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
    } else {
        // glBegin(GL_QUADS);
        // glTexCoord2f(0.0, 1.0);
        // glVertex2f(-1.0, -1.0);
        // glTexCoord2f(1.0, 1.0);
        // glVertex2f(1.0, -1.0);
        // glTexCoord2f(1.0, 0.0);
        // glVertex2f(1.0, 1.0);
        // glTexCoord2f(0.0, 0.0);
        // glVertex2f(-1.0, 1.0);
        // glEnd();

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
    _canvas->get_color_attach_texture()->unbind();

    //render navigator
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
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, _height, _width, 0, 0, 0, _width, _height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);

    CHECK_GL_ERROR;

    // CHECK_GL_ERROR;
    // {
    //     if (_canvas->get_color_attach_texture(1)) {
    //         std::stringstream ss;
    //         ss << "/home/wangrui22/data/output_para_" << _width << "_" << _height << ".raw";
    //         _canvas->debug_output_color_1(ss.str());
    //         _canvas->debug_output_color_0(ss.str() + "_color");
    //     }
    // }
    // CHECK_GL_ERROR;

    // Test code
    // {
    //         FBOStack fbo_stack;
    //         _scene_color_attach_0->bind();
    //         unsigned char *color_array = new unsigned char[_width * _height * 3];
    //         _scene_color_attach_0->download(GL_RGB, GL_UNSIGNED_BYTE,
    //         color_array);
    //     #ifdef WIN32
    //         std::stringstream ss;
    //         ss << "D:/temp/scene_img_" << _name << "_" << _width << "_" <<
    //         _height
    //            << ".rgb";
    //         FileUtil::write_raw(ss.str(), (char *)color_array, _width * _height *
    //         3);
    //     #else
    //         std::stringstream ss;
    //         ss << "/home/wangrui22/data/scene_img_" << _name << "_" << _width << "_" <<
    //         _height
    //            << ".rgb";
    //         FileUtil::write_raw(ss.str(), (char *)color_array, _width * _height *
    //         3);
    //     #endif
    // }


    set_dirty(false);
    MI_RENDERALGO_LOG(MI_TRACE) << "OUT ray cast scene render.";
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

        if (GPU == Configure::instance()->get_processing_unit_type()) {
            // set texture
            _ray_caster->set_pseudo_color_texture(_pseudo_color_texture,
                                                  S_TRANSFER_FUNC_WIDTH);
            _ray_caster->set_color_opacity_texture_array(
                _color_opacity_texture_array);
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

    if (GPU == Configure::instance()->get_processing_unit_type()) {
        const int tex_num = static_cast<int>(label_level);
        unsigned char* rgba =
            new unsigned char[S_TRANSFER_FUNC_WIDTH * tex_num * 4];
        memset(rgba, 0, S_TRANSFER_FUNC_WIDTH * tex_num * 4);

        // reshape color opacity texture array
        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array, GL_CLAMP_TO_EDGE,
            GL_LINEAR, GL_RGBA8, S_TRANSFER_FUNC_WIDTH, tex_num, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, (char*)rgba);
    }

    set_dirty(true);
}

void RayCastScene::set_sample_rate(float sample_rate) {
    _ray_caster->set_sample_rate(sample_rate);
    set_dirty(true);
}

void RayCastScene::set_visible_labels(std::vector<unsigned char> labels) {
    for (auto it = labels.begin(); it != labels.end(); ++it) {
        if (*it == 0) {
            RENDERALGO_THROW_EXCEPTION("visible labels contain zero");
        }
    }

    if (_volume_infos && !labels.empty()) {
        //here can't update empty visibile labels(none-mask)
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

int RayCastScene::get_window_level(float& ww, float& wl,
                                   unsigned char label) const {
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
    if (GPU == Configure::instance()->get_processing_unit_type()) {
        RENDERALGO_CHECK_NULL_EXCEPTION(_pseudo_color_texture);

        std::vector<ColorTFPoint> pts;
        color->set_width(S_TRANSFER_FUNC_WIDTH);
        color->get_point_list(pts);
        unsigned char* rgb = new unsigned char[S_TRANSFER_FUNC_WIDTH * 3];

        for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
            rgb[i * 3] = static_cast<unsigned char>(pts[i].x);
            rgb[i * 3 + 1] = static_cast<unsigned char>(pts[i].y);
            rgb[i * 3 + 2] = static_cast<unsigned char>(pts[i].z);
        }

        GLTextureCache::instance()->cache_update(
            GL_TEXTURE_1D, _pseudo_color_texture, 0, 0, 0, S_TRANSFER_FUNC_WIDTH, 0,
            0, GL_RGB, GL_UNSIGNED_BYTE, (char*)rgb);
    }

    set_dirty(true);
}

void RayCastScene::set_color_opacity(std::shared_ptr<ColorTransFunc> color,
                                     std::shared_ptr<OpacityTransFunc> opacity,
                                     unsigned char label) {
    if (GPU == Configure::instance()->get_processing_unit_type()) {
        std::vector<ColorTFPoint> color_pts;
        color->set_width(S_TRANSFER_FUNC_WIDTH);
        color->get_point_list(color_pts);

        std::vector<OpacityTFPoint> opacity_pts;
        opacity->set_width(S_TRANSFER_FUNC_WIDTH);
        opacity->get_point_list(opacity_pts);

        unsigned char* rgba = new unsigned char[S_TRANSFER_FUNC_WIDTH * 4];

        for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
            rgba[i * 4] = static_cast<unsigned char>(color_pts[i].x);
            rgba[i * 4 + 1] = static_cast<unsigned char>(color_pts[i].y);
            rgba[i * 4 + 2] = static_cast<unsigned char>(color_pts[i].z);
            rgba[i * 4 + 3] = static_cast<unsigned char>(opacity_pts[i].a);
        }

        GLTextureCache::instance()->cache_update(
            GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array, 0, label, 0,
            S_TRANSFER_FUNC_WIDTH, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, (char*)rgba);
    }

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