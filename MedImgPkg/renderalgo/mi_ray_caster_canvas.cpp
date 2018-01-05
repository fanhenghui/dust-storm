#include "mi_ray_caster_canvas.h"

#include "util/mi_file_util.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"

#include "cudaresource/mi_cuda_device_memory.h"
#include "cudaresource/mi_cuda_resource_manager.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

RayCasterCanvas::RayCasterCanvas(RayCastingStrategy strategy, GPUPlatform p)
    : _stratrgy(strategy), _gpu_platform(p), 
      _has_init(false), _width(32), _height(32) {}

RayCasterCanvas::~RayCasterCanvas() {}

void RayCasterCanvas::initialize(bool multi_color_attach) {
    if (_has_init) {
        return;
    }

    if (CPU_BASE == _stratrgy) {
        // 1 Create color array for CPU writing
        _color_array.reset(new RGBAUnit[_width * _height]);
        // 2 Create a texture for mapping in ray cast scene
        UIDType texture_color_id = 0;
        GLTexture2DPtr gl_color_attach_0 = GLResourceManagerContainer::instance()
            ->get_texture_2d_manager()->create_object(texture_color_id);
        _color_attach_0.reset(new GPUCanvasPair(gl_color_attach_0));

        gl_color_attach_0->set_description("ray caster canvas FBO color attachment 0 texture");
        gl_color_attach_0->initialize();
        gl_color_attach_0->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        gl_color_attach_0->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        _res_shield.add_shield<GLTexture2D>(gl_color_attach_0);

    } else {
        if (GPU_BASE == _gpu_platform) {
            //FBO with 2 color attachment and 1 depth attachment
            CHECK_GL_ERROR

            UIDType fbo_id = 0;
            _gl_fbo = GLResourceManagerContainer::instance()
                ->get_fbo_manager()->create_object(fbo_id);
            _gl_fbo->set_description("ray caster canvas FBO");
            _gl_fbo->initialize();
            _gl_fbo->set_target(GL_FRAMEBUFFER);

            UIDType texture_color_id = 0;
            GLTexture2DPtr gl_color_attach_0 = GLResourceManagerContainer::instance()
                ->get_texture_2d_manager()->create_object(texture_color_id);
            _color_attach_0.reset(new GPUCanvasPair(gl_color_attach_0));
            gl_color_attach_0->set_description("ray caster canvas FBO color attachment 0 texture");
            gl_color_attach_0->initialize();
            gl_color_attach_0->bind();
            GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
            GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
            gl_color_attach_0->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

            if (multi_color_attach) {
                GLTexture2DPtr gl_color_attach_1 = GLResourceManagerContainer::instance()
                    ->get_texture_2d_manager()->create_object(texture_color_id);
                _color_attach_1.reset(new GPUCanvasPair(gl_color_attach_1));
                gl_color_attach_1->set_description("ray caster canvas FBO color attachment 1 texture");
                gl_color_attach_1->initialize();
                gl_color_attach_1->bind();
                GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
                GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
                gl_color_attach_1->load(GL_RGB8, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            }

            UIDType depth_color_id = 0;
            _gl_depth_attach = GLResourceManagerContainer::instance()
                ->get_texture_2d_manager()->create_object(depth_color_id);
            _gl_depth_attach->set_description("ray caster canvas FBO depth attachment texture");
            _gl_depth_attach->initialize();
            _gl_depth_attach->bind();
            GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
            GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
            _gl_depth_attach->load(GL_DEPTH_COMPONENT16, _width, _height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);

            // bind texture to FBO
            _gl_fbo->bind();
            _gl_fbo->attach_texture(GL_COLOR_ATTACHMENT0, gl_color_attach_0);
            _gl_fbo->attach_texture(GL_DEPTH_ATTACHMENT, _gl_depth_attach);
            if (multi_color_attach) {
                _gl_fbo->attach_texture(GL_COLOR_ATTACHMENT1, _color_attach_1->get_gl_resource());
            }

            _gl_fbo->unbind();

            CHECK_GL_ERROR;

            _res_shield.add_shield<GLFBO>(_gl_fbo);
            _res_shield.add_shield<GLTexture2D>(gl_color_attach_0);
            _res_shield.add_shield<GLTexture2D>(_gl_depth_attach);
            if (multi_color_attach) {
                _res_shield.add_shield<GLTexture2D>(_color_attach_1->get_gl_resource());
            }
        } else {
            //two device memory as canvas color attachment
            CudaDeviceMemoryPtr cuda_mem_0 = CudaResourceManager::instance()->create_device_memory("ray caster canvas color-0");
            cuda_mem_0->load(_width*_height * 4, nullptr);
            _color_attach_0.reset(new GPUCanvasPair(cuda_mem_0));
            
            if (multi_color_attach) {
                CudaDeviceMemoryPtr cuda_mem_1 = CudaResourceManager::instance()->create_device_memory("ray caster canvas color-1");
                cuda_mem_1->load(_width*_height * 4, nullptr);
                _color_attach_1.reset(new GPUCanvasPair(cuda_mem_1));
            }
        }
    }

    _has_init = true;
}

void RayCasterCanvas::set_display_size(int width, int height) {
    _width = width;
    _height = height;
    _color_array.reset(new RGBAUnit[_width * _height]);

    if (_has_init) {
        if (CPU_BASE == _stratrgy) {
            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_2D, _color_attach_0->get_gl_resource(), GL_CLAMP_TO_EDGE, GL_LINEAR, GL_RGBA8,
                _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }
        else {
            if (GL_BASE == _gpu_platform) {
                GLTextureCache::instance()->cache_load(
                    GL_TEXTURE_2D, _color_attach_0->get_gl_resource(), GL_CLAMP_TO_EDGE, GL_LINEAR, GL_RGBA8,
                    _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                GLTextureCache::instance()->cache_load(
                    GL_TEXTURE_2D, _gl_depth_attach, GL_CLAMP_TO_EDGE, GL_LINEAR,GL_DEPTH_COMPONENT16, 
                    _width, _height, 0, GL_DEPTH_COMPONENT,GL_UNSIGNED_SHORT, nullptr);

                if (_color_attach_1) {
                    GLTextureCache::instance()->cache_load(
                        GL_TEXTURE_2D, _color_attach_1->get_gl_resource(), GL_CLAMP_TO_EDGE, GL_LINEAR, GL_RGB8,
                        _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
                }
            } else {
                _color_attach_0->get_cuda_resource()->load(_width*_height*4, nullptr);
                if (_color_attach_1) {
                    _color_attach_1->get_cuda_resource()->load(_width*_height * 4, nullptr);
                }
            }
        }
    }        
}

GLFBOPtr RayCasterCanvas::get_fbo() {
    return _gl_fbo;
}

RGBAUnit* RayCasterCanvas::get_color_array() {
    return _color_array.get();
}

void RayCasterCanvas::update_color_array() {
    if (CPU_BASE == _stratrgy) {
        CHECK_GL_ERROR;
        _color_attach_0->get_gl_resource()->bind();
        _color_attach_0->get_gl_resource()->update(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, _color_array.get());
        CHECK_GL_ERROR;
    } else {
        MI_RENDERALGO_LOG(MI_WARNING) << "call update color array in GPU canvas.";
    }
}

GPUCanvasPairPtr RayCasterCanvas::get_color_attach_texture(int id/*=0*/) {
    if (id == 0) {
        return _color_attach_0;
    } else if (id == 1) {
        return _color_attach_1;
    } else {
        RENDERALGO_THROW_EXCEPTION("invalid id to get color attach texture.");
    }
}

void RayCasterCanvas::debug_output_color_0(const std::string& file_name) {
    std::unique_ptr<unsigned char[]> color_array(new unsigned char[_width * _height * 3]);
    if (CPU_BASE == _stratrgy || (GPU_BASE == _stratrgy && GL_BASE == _gpu_platform)) {
        _color_attach_0->get_gl_resource()->bind();
        _color_attach_0->get_gl_resource()->download(GL_RGB, GL_UNSIGNED_BYTE, color_array.get());
        _color_attach_0->get_gl_resource()->unbind();
    } else {
        std::unique_ptr<unsigned char[]> rgba(new unsigned char[_width * _height * 4]);
        _color_attach_0->get_cuda_resource()->download(rgba.get(), _width*_height*4);
        for (int i = 0; i < _width*_height; ++i) {
            color_array[i * 3 + 0] = rgba[i * 4 + 0];
            color_array[i * 3 + 1] = rgba[i * 4 + 1];
            color_array[i * 3 + 2] = rgba[i * 4 + 2];
        }
    }
    FileUtil::write_raw(file_name, color_array.get(), _width*_height*3);
}

void RayCasterCanvas::debug_output_color_1(const std::string& file_name) {
    if (CPU_BASE == _stratrgy) {
        return;
    }
    
    if (nullptr == _color_attach_1) {
        MI_RENDERALGO_LOG(MI_DEBUG) << "rc canvas attachment 1 is null.";
        return;
    }

    std::unique_ptr<unsigned char[]> color_array(new unsigned char[_width * _height * 3]);
    if (GL_BASE == _gpu_platform) {
        _color_attach_1->get_gl_resource()->bind();
        _color_attach_1->get_gl_resource()->download(GL_RGB, GL_UNSIGNED_BYTE, color_array.get());
        _color_attach_1->get_gl_resource()->unbind();
    } else {
        std::unique_ptr<unsigned char[]> rgba(new unsigned char[_width * _height * 4]);
        _color_attach_1->get_cuda_resource()->download(rgba.get(), _width*_height * 4);
        for (int i = 0; i < _width*_height; ++i) {
            color_array[i * 3 + 0] = rgba[i * 4 + 0];
            color_array[i * 3 + 1] = rgba[i * 4 + 1];
            color_array[i * 3 + 2] = rgba[i * 4 + 2];
        }
    }
    FileUtil::write_raw(file_name, color_array.get(), _width*_height * 3);
}

void RayCasterCanvas::get_display_size(int& width, int& height) const {
    width = _width;
    height = _height;
}

MED_IMG_END_NAMESPACE