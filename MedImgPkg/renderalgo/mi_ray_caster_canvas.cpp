#include "mi_ray_caster_canvas.h"
#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

RayCasterCanvas::RayCasterCanvas()
    : _has_init(false), _width(32), _height(32) {}

RayCasterCanvas::~RayCasterCanvas() {}

void RayCasterCanvas::initialize() {
    if (!_has_init) {
        CHECK_GL_ERROR

        UIDType fbo_id = 0;
        _gl_fbo = GLResourceManagerContainer::instance()
                  ->get_fbo_manager()
                  ->create_object(fbo_id);
        _gl_fbo->set_description("ray caster canvas FBO");
        _gl_fbo->initialize();
        _gl_fbo->set_target(GL_FRAMEBUFFER);

        UIDType texture_color_id = 0;
        _color_attach_0 = GLResourceManagerContainer::instance()
                          ->get_texture_2d_manager()
                          ->create_object(texture_color_id);
        _color_attach_0->set_description(
            "ray caster canvas FBO color attachment 0 texture");
        _color_attach_0->initialize();
        _color_attach_0->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _color_attach_0->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE,
                              nullptr);

        UIDType depth_color_id = 0;
        _depth_attach = GLResourceManagerContainer::instance()
                        ->get_texture_2d_manager()
                        ->create_object(depth_color_id);
        _depth_attach->set_description(
            "ray caster canvas FBO depth attachment texture");
        _depth_attach->initialize();
        _depth_attach->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _depth_attach->load(GL_DEPTH_COMPONENT16, _width, _height,
                            GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);

        // bind texture to FBO
        _gl_fbo->bind();

        _gl_fbo->attach_texture(GL_COLOR_ATTACHMENT0, _color_attach_0);
        _gl_fbo->attach_texture(GL_DEPTH_ATTACHMENT, _depth_attach);

        _gl_fbo->unbind();

        CHECK_GL_ERROR;

        // Create array
        _color_array.reset(new RGBAUnit[_width * _height]);

        _has_init = true;

        _res_shield.add_shield<GLFBO>(_gl_fbo);
        _res_shield.add_shield<GLTexture2D>(_color_attach_0);
        _res_shield.add_shield<GLTexture2D>(_depth_attach);
    }
}

void RayCasterCanvas::set_display_size(int width, int height) {
    _width = width;
    _height = height;
    _color_array.reset(new RGBAUnit[_width * _height]);

    if (_has_init) {

        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_2D, _color_attach_0, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_RGBA8,
            _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_2D, _depth_attach, GL_CLAMP_TO_EDGE, GL_LINEAR,
            GL_DEPTH_COMPONENT16, _width, _height, 0, GL_DEPTH_COMPONENT,
            GL_UNSIGNED_SHORT, nullptr);

        // _color_attach_0->bind();
        // _color_attach_0->load(GL_RGBA8, _width, _height, GL_RGBA,
        // GL_UNSIGNED_BYTE,
        //                       nullptr);

        // _depth_attach->bind();
        // _depth_attach->load(GL_DEPTH_COMPONENT16, _width, _height,
        //                     GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);
    }
}

GLFBOPtr RayCasterCanvas::get_fbo() {
    return _gl_fbo;
}

RGBAUnit* RayCasterCanvas::get_color_array() {
    return _color_array.get();
}

void RayCasterCanvas::update_color_array() {
    CHECK_GL_ERROR
    _color_attach_0->bind();
    _color_attach_0->update(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE,
                            _color_array.get());
    CHECK_GL_ERROR
}

GLTexture2DPtr RayCasterCanvas::get_color_attach_texture() {
    return _color_attach_0;
}

void RayCasterCanvas::debug_output_color(const std::string& file_name) {
    _color_attach_0->bind();
    std::unique_ptr<unsigned char[]> color_array(
        new unsigned char[_width * _height * 4]);
    _color_attach_0->download(GL_RGBA, GL_UNSIGNED_BYTE, color_array.get());

    std::ofstream out(file_name, std::ios::out | std::ios::binary);

    if (out.is_open()) {
        out.write((char*)color_array.get(), _width * _height * 4);
    }

    out.close();
}

void RayCasterCanvas::get_display_size(int& width, int& height) const {
    width = _width;
    height = _height;
}

MED_IMG_END_NAMESPACE