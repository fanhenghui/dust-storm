#include "mi_scene_base.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"
#include "util/mi_file_util.h"

// #include "libgpujpeg/gpujpeg_encoder.h"
// #include "libgpujpeg/gpujpeg_encoder_internal.h"

MED_IMG_BEGIN_NAMESPACE

SceneBase::SceneBase() : _width(128), _height(128) {
    _image_buffer[0].reset(new unsigned char[_width * _height * 3]);
    _image_buffer[1].reset(new unsigned char[_width * _height * 3]);
    _image_buffer_size[0] = _width * _height * 3;
    _image_buffer_size[1] = _width * _height * 3;

    _dirty = true;
    _name = "Scene";
    _front_buffer_id = 0;

#ifndef WIN32
    // init gpujepg parameter
    _gpujpeg_encoder = nullptr;
    _gpujpeg_texture = nullptr;
    _gpujpeg_encoder_dirty = false;
    //_gpujpeg_encoding_duration = 0;
#endif
}

SceneBase::SceneBase(int width, int height) : _width(width), _height(height) {
    _image_buffer[0].reset(new unsigned char[_width * _height * 3]);
    _image_buffer[1].reset(new unsigned char[_width * _height * 3]);
    _image_buffer_size[0] = _width * _height * 3;
    _image_buffer_size[1] = _width * _height * 3;

    _dirty = true;
    _name = "Scene";
    _front_buffer_id = 0;

#ifndef WIN32
    // init gpujepg parameter
    _gpujpeg_encoder = nullptr;
    _gpujpeg_texture = nullptr;
    _gpujpeg_encoder_dirty = false;
    //_gpujpeg_encoding_duration = 0;
#endif
}

SceneBase::~SceneBase() {}

void SceneBase::render_to_back() {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _scene_fbo->get_id());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glDrawBuffer(GL_BACK);
    glBlitFramebuffer(0, _height, _width, 0, 0, 0, _width, _height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST); // flip vertically copy
}

std::shared_ptr<CameraBase> SceneBase::get_camera() {
    return _camera;
}

void SceneBase::initialize() {
    if (!_scene_fbo) {
        // Init FBO
        CHECK_GL_ERROR;

        GLUtils::set_pixel_pack_alignment(1);

        UIDType fbo_id = 0;
        _scene_fbo = GLResourceManagerContainer::instance()
                     ->get_fbo_manager()
                     ->create_object(fbo_id);
        _scene_fbo->set_description("scene base FBO");
        _scene_fbo->initialize();
        _scene_fbo->set_target(GL_FRAMEBUFFER);

        UIDType texture_color_id = 0;
        _scene_color_attach_0 = GLResourceManagerContainer::instance()
                                ->get_texture_2d_manager()
                                ->create_object(texture_color_id);
        _scene_color_attach_0->set_description("scene base color attachment 0");
        _scene_color_attach_0->initialize();
        _scene_color_attach_0->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _scene_color_attach_0->load(GL_RGB8, _width, _height, GL_RGB,
                                    GL_UNSIGNED_BYTE, nullptr);

        UIDType depth_color_id = 0;
        _scene_depth_attach = GLResourceManagerContainer::instance()
                              ->get_texture_2d_manager()
                              ->create_object(depth_color_id);
        _scene_depth_attach->set_description("scene base depth attachment");
        _scene_depth_attach->initialize();
        _scene_depth_attach->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _scene_depth_attach->load(GL_DEPTH_COMPONENT16, _width, _height,
                                  GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);

        // bind texture to FBO
        _scene_fbo->bind();

        _scene_fbo->attach_texture(GL_COLOR_ATTACHMENT0, _scene_color_attach_0);
        _scene_fbo->attach_texture(GL_DEPTH_ATTACHMENT, _scene_depth_attach);

        _scene_fbo->unbind();

        CHECK_GL_ERROR;

#ifndef WIN32
        // init gpujpeg device(TODO multi-gpu situation!!!!!!!! especially in
        // multi-scene)
        gpujpeg_init_device(0, 0);

        gpujpeg_set_default_parameters(&_gpujpeg_param);        //默认参数
        gpujpeg_parameters_chroma_subsampling(&_gpujpeg_param); //默认采样参数;
        //_gpujpeg_param.quality = 98;

        gpujpeg_image_set_default_parameters(&_gpujpeg_image_param);
        _gpujpeg_image_param.width = _width;
        _gpujpeg_image_param.height = _height;
        _gpujpeg_image_param.comp_count = 3;
        _gpujpeg_image_param.color_space = GPUJPEG_RGB;
        _gpujpeg_image_param.sampling_factor = GPUJPEG_4_4_4;

        // bind GL texture to cuda(by PBO)
        unsigned int tex_id = _scene_color_attach_0->get_id();
        _gpujpeg_texture =
            gpujpeg_opengl_texture_register(tex_id, GPUJPEG_OPENGL_TEXTURE_READ);
        // create encoder
        _gpujpeg_encoder =
            gpujpeg_encoder_create(&_gpujpeg_param, &_gpujpeg_image_param);
        RENDERALGO_CHECK_NULL_EXCEPTION(_gpujpeg_encoder);
        // set texture as input
        gpujpeg_encoder_input_set_texture(&_gpujpeg_encoder_input,
                                          _gpujpeg_texture);

        // cudaEventCreate(&_gpujpeg_encoding_start);
        // cudaEventCreate(&_gpujpeg_encoding_stop);
#endif

        _res_shield.add_shield<GLFBO>(_scene_fbo);
        _res_shield.add_shield<GLTexture2D>(_scene_color_attach_0);
        _res_shield.add_shield<GLTexture2D>(_scene_depth_attach);
    }
}

void SceneBase::set_display_size(int width, int height) {
    _width = width;
    _height = height;

    if (!_scene_fbo) {
        return;
    }

    // reset image buffer
    _image_buffer[0].reset(new unsigned char[_width * _height * 3]);
    _image_buffer[1].reset(new unsigned char[_width * _height * 3]);
    _image_buffer_size[0] = _width * _height * 3;
    _image_buffer_size[1] = _width * _height * 3;
    _front_buffer_id = 0;

    GLTextureCache::instance()->cache_load(
        GL_TEXTURE_2D, _scene_color_attach_0, GL_CLAMP_TO_EDGE, GL_LINEAR,
        GL_RGB8, _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    GLTextureCache::instance()->cache_load(
        GL_TEXTURE_2D, _scene_depth_attach, GL_CLAMP_TO_EDGE, GL_LINEAR,
        GL_DEPTH_COMPONENT16, _width, _height, 0, GL_DEPTH_COMPONENT,
        GL_UNSIGNED_SHORT, nullptr);

#ifndef WIN32
    _gpujpeg_image_param.width = _width;
    _gpujpeg_image_param.height = _height;
    _gpujpeg_encoder_dirty = true;
#endif
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

void SceneBase::pre_render_i() {
#ifndef WIN32
    if (_gpujpeg_encoder && _gpujpeg_encoder_dirty) {
        gpujpeg_encoder_destroy(_gpujpeg_encoder);
        _gpujpeg_encoder = nullptr;

        gpujpeg_image_set_default_parameters(&_gpujpeg_image_param);
        _gpujpeg_image_param.width = _width;
        _gpujpeg_image_param.height = _height;
        _gpujpeg_image_param.comp_count = 3;
        _gpujpeg_image_param.color_space = GPUJPEG_RGB;
        _gpujpeg_image_param.sampling_factor = GPUJPEG_4_4_4;

        // bind GL texture to cuda(by PBO)
        unsigned int tex_id = _scene_color_attach_0->get_id();
        _gpujpeg_texture =
            gpujpeg_opengl_texture_register(tex_id, GPUJPEG_OPENGL_TEXTURE_READ);
        // create encoder
        _gpujpeg_encoder =
            gpujpeg_encoder_create(&_gpujpeg_param, &_gpujpeg_image_param);
        RENDERALGO_CHECK_NULL_EXCEPTION(_gpujpeg_encoder);
        // set texture as input
        gpujpeg_encoder_input_set_texture(&_gpujpeg_encoder_input,
                                          _gpujpeg_texture);

        _gpujpeg_encoder_dirty = false;
    }
#endif
}

void SceneBase::set_name(const std::string& name) {
    _name = name;
}

const std::string& SceneBase::get_name() const {
    return _name;
}

void SceneBase::download_image_buffer(bool jpeg /*= true*/) {
    boost::mutex::scoped_lock locker(_write_mutex);
#ifndef WIN32
    if (jpeg) 
    {
        uint8_t* image_compressed = nullptr;
        int image_compressed_size = 0;

        // Record cuda time of encoding(在OpenGL的环境下时间不对,得加一个glFinish)
        //::glFinish();
        // cudaEventRecord(_gpujpeg_encoding_start,0);
        CHECK_GL_ERROR;

        int err = gpujpeg_encoder_encode(_gpujpeg_encoder, &_gpujpeg_encoder_input,
                                         &image_compressed, &image_compressed_size);

        CHECK_GL_ERROR;

        if (err != 0) {
            RENDERALGO_THROW_EXCEPTION("GPU jpeg encoding failed!");
        }

        // cudaEventRecord(_gpujpeg_encoding_stop,0);
        // cudaEventSynchronize(_gpujpeg_encoding_stop);
        // cudaEventElapsedTime(&_gpujpeg_encoding_duration ,_gpujpeg_encoding_start
        // , _gpujpeg_encoding_stop);

        // std::cout << "encoding spec : \n";
        // std::cout << " memory to : " <<
        // _gpujpeg_encoder->coder.duration_memory_to << std::endl;
        // std::cout << " memory from : " <<
        // _gpujpeg_encoder->coder.duration_memory_from << std::endl;
        // std::cout << " memory map : " <<
        // _gpujpeg_encoder->coder.duration_memory_map << std::endl;
        // std::cout << " memory unmap : " <<
        // _gpujpeg_encoder->coder.duration_memory_unmap << std::endl;
        // std::cout << " preprocessor : " <<
        // _gpujpeg_encoder->coder.duration_preprocessor << std::endl;
        // std::cout << " dct_quantization : " <<
        // _gpujpeg_encoder->coder.duration_dct_quantization << std::endl;
        // std::cout << " huffman coder : " <<
        // _gpujpeg_encoder->coder.duration_huffman_coder << std::endl;
        // std::cout << " stream : " << _gpujpeg_encoder->coder.duration_stream <<
        // std::endl;
        // std::cout << " in gpu : " << _gpujpeg_encoder->coder.duration_in_gpu <<
        // std::endl;
        // std::cout << " jpeg encoding time : " << _gpujpeg_encoding_duration <<
        // std::endl;
        // std::cout << std::endl;

        // copy image_compressed to image_buffer
        memcpy((char*)(_image_buffer[1 - _front_buffer_id].get()),
               image_compressed, image_compressed_size);
        _image_buffer_size[1 - _front_buffer_id] = image_compressed_size;

        // FileUtil::write_raw("/home/wr/data/output_download.jpeg",_image_buffer[1
        // - _front_buffer_id].get() , image_compressed_size);
    }
    else 
#endif
    {
        // download FBO to back buffer directly
        CHECK_GL_ERROR;

        _scene_color_attach_0->bind();
        _scene_color_attach_0->download(GL_RGB, GL_UNSIGNED_BYTE,
                                        _image_buffer[1 - _front_buffer_id].get());
        _image_buffer_size[1 - _front_buffer_id] = _width * _height * 3;

        CHECK_GL_ERROR;

        // FileUtil::write_raw("/home/wr/data/output_download.raw",_image_buffer[1 -
        // _front_buffer_id].get() , _width*_height*4);
    }
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

// float SceneBase::get_compressing_time() const
//{
//    return _gpujpeg_encoding_duration;
//}

MED_IMG_END_NAMESPACE