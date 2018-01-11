#include "mi_scene_base.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"
#include "util/mi_file_util.h"

#include "mi_gpu_image_compressor.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

SceneBase::SceneBase(GPUPlatform platfrom) :
    _width(128), _height(128), _gpu_platform(platfrom) {
    _image_buffer[0].reset(new unsigned char[_width * _height * 3]);
    _image_buffer[1].reset(new unsigned char[_width * _height * 3]);
    _image_buffer_size[0] = _width * _height * 3;
    _image_buffer_size[1] = _width * _height * 3;

    _dirty = true;
    _name = "Scene";
    _front_buffer_id = 0;
    _downsample = false;  
    _compress_hd_quality = 80;
    _compress_ld_quality  = 15;
    _gpujpeg_encoder_dirty = false;

    _compressor.reset(new GPUImgCompressor(platfrom));
}

SceneBase::SceneBase(GPUPlatform platfrom, int width, int height) :
    _width(width), _height(height), _gpu_platform(platfrom) {
    _image_buffer[0].reset(new unsigned char[_width * _height * 3]);
    _image_buffer[1].reset(new unsigned char[_width * _height * 3]);
    _image_buffer_size[0] = _width * _height * 3;
    _image_buffer_size[1] = _width * _height * 3;

    _dirty = true;
    _name = "Scene";
    _front_buffer_id = 0;
    _downsample = false;
    _compress_hd_quality = 80;
    _compress_ld_quality  = 15;
    _gpujpeg_encoder_dirty = false;

    _compressor.reset(new GPUImgCompressor(platfrom));
}

SceneBase::~SceneBase() {}

void SceneBase::render_to_back() {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _scene_fbo->get_id());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glDrawBuffer(GL_BACK);
    glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

std::shared_ptr<CameraBase> SceneBase::get_camera() {
    return _camera;
}

void SceneBase::initialize() {
    //----------------------------------------------------//
    //create FBO (if no need to create, please rewrite this interface)
    //----------------------------------------------------//
    if (!_scene_fbo) {
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

        _scene_color_attach_1 = GLResourceManagerContainer::instance()
            ->get_texture_2d_manager()->create_object("scene base color attachment 1 <flip verticalily>");
        _scene_color_attach_1->initialize();
        _scene_color_attach_1->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _scene_color_attach_1->load(GL_RGB8, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

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
        _scene_fbo->attach_texture(GL_COLOR_ATTACHMENT1, _scene_color_attach_1);
        _scene_fbo->attach_texture(GL_DEPTH_ATTACHMENT, _scene_depth_attach);
        _scene_fbo->unbind();

        CHECK_GL_ERROR;

        _res_shield.add_shield<GLFBO>(_scene_fbo);
        _res_shield.add_shield<GLTexture2D>(_scene_color_attach_0);
        _res_shield.add_shield<GLTexture2D>(_scene_color_attach_1);
        _res_shield.add_shield<GLTexture2D>(_scene_depth_attach);

        //-------------------------------//
        // GPU compressor
        //-------------------------------//
        std::vector<int> qualitys(2);
        qualitys[0] = _compress_ld_quality;
        qualitys[1] = _compress_hd_quality;
        _compressor->set_image(GPUCanvasPairPtr(new GPUCanvasPair(_scene_color_attach_1)), qualitys);
    }
}

void SceneBase::set_display_size(int width, int height) {
    if (width == _width && height == _height) {
        return;
    }

    _width = width;
    _height = height;

    // reset image buffer
    _image_buffer[0].reset(new unsigned char[_width * _height * 3]);
    _image_buffer[1].reset(new unsigned char[_width * _height * 3]);
    _image_buffer_size[0] = _width * _height * 3;
    _image_buffer_size[1] = _width * _height * 3;
    _front_buffer_id = 0;

    if (_scene_fbo) {
        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_2D, _scene_color_attach_0, GL_CLAMP_TO_EDGE, GL_LINEAR,
            GL_RGB8, _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_2D, _scene_depth_attach, GL_CLAMP_TO_EDGE, GL_LINEAR,
            GL_DEPTH_COMPONENT16, _width, _height, 0, GL_DEPTH_COMPONENT,
            GL_UNSIGNED_SHORT, nullptr);

        //color-attach-1 may be remove if use cuda platform do compress
        if (_scene_color_attach_1) {
            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_2D, _scene_color_attach_1, GL_CLAMP_TO_EDGE, GL_LINEAR,
                GL_RGB8, _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        }
    }

    _gpujpeg_encoder_dirty = true;

    set_dirty(true);
}

void SceneBase::render() {}

void SceneBase::rotate(const Point2& pre_pt, const Point2& cur_pt) {}

void SceneBase::zoom(const Point2& pre_pt, const Point2& cur_pt) {}

void SceneBase::pan(const Point2& pre_pt, const Point2& cur_pt) {}

void SceneBase::get_display_size(int& width, int& height) const {
    width = _width;
    height = _height;
}

void SceneBase::set_dirty(bool flag) {
    _dirty = flag;
}

bool SceneBase::get_dirty() const {
    return _dirty;
}

void SceneBase::pre_render() {
    if (_gpujpeg_encoder_dirty) {
        std::vector<int> qualitys(2);
        qualitys[0] = _compress_ld_quality;
        qualitys[1] = _compress_hd_quality;
        _compressor->set_image(GPUCanvasPairPtr(new GPUCanvasPair(_scene_color_attach_1)), qualitys);
        _gpujpeg_encoder_dirty = false;
    }
}

void SceneBase::set_name(const std::string& name) {
    _name = name;
}

const std::string& SceneBase::get_name() const {
    return _name;
}

void SceneBase::download_image_buffer(bool jpeg /*= true*/) {
    boost::mutex::scoped_lock locker(_write_mutex);

    if (jpeg)  {
        int compressed_size = 0;
        int err = 0;
        if (_downsample) {
            err = _compressor->compress(_compress_ld_quality, (char*)(_image_buffer[1 - _front_buffer_id].get()), compressed_size);
        } else {
            err = _compressor->compress(_compress_hd_quality, (char*)(_image_buffer[1 - _front_buffer_id].get()), compressed_size);
        }
        _image_buffer_size[1 - _front_buffer_id] = compressed_size;

        if (err != 0) {
            MI_RENDERALGO_LOG(MI_ERROR) << "scene " << this->get_name() << " download image failed.";
        }
    }
    else {
        // download FBO to back buffer directly
        CHECK_GL_ERROR;

        _scene_color_attach_0->bind();
        _scene_color_attach_0->download(GL_RGB, GL_UNSIGNED_BYTE, _image_buffer[1 - _front_buffer_id].get());
        _image_buffer_size[1 - _front_buffer_id] = _width * _height * 3;

        CHECK_GL_ERROR;
    }
    // FileUtil::write_raw("/home/wr/data/output_download.raw",_image_buffer[1 -
    // _front_buffer_id].get() , _width*_height*4);
}

void SceneBase::swap_image_buffer() {
    boost::mutex::scoped_lock locker0(_read_mutex);
    boost::mutex::scoped_lock locker1(_write_mutex);
    _front_buffer_id = 1 - _front_buffer_id;
}

void SceneBase::get_image_buffer(unsigned char*& buffer, int& size) {
    boost::mutex::scoped_lock locker(_read_mutex);
    buffer = _image_buffer[_front_buffer_id].get();
    size = _image_buffer_size[_front_buffer_id];
}

void SceneBase::set_downsample(bool flag) {
    _downsample = flag;
}

bool SceneBase::get_downsample() const {
    return _downsample;
}

float SceneBase::get_compressing_duration() const
{
    return _compressor->get_last_duration();
}

void SceneBase::set_compress_hd_quality(int quality) {
    _compress_hd_quality = quality;
}

void SceneBase::set_compress_ld_quality(int quality) {
    _compress_ld_quality  = quality;
}

MED_IMG_END_NAMESPACE