#include "mi_entry_exit_points.h"

#include "util/mi_file_util.h"

#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"

#include "cudaresource/mi_cuda_surface_2d.h"
#include "cudaresource/mi_cuda_resource_manager.h"

#include "io/mi_image_data.h"

#include "arithmetic/mi_camera_base.h"

#include "mi_camera_calculator.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

EntryExitPoints::EntryExitPoints(RayCastingStrategy s, GPUPlatform p)
    : _strategy(s), _gpu_platform(p), _width(4), _height(4), _has_init(false) {
    if (CPU_BASE == _strategy) {
        _entry_points_buffer.reset(new Vector4f[_width * _height]);
        _exit_points_buffer.reset(new Vector4f[_width * _height]);
    } else {
        if (GL_BASE == _gpu_platform) {
            GLTexture2DPtr entry_tex = GLResourceManagerContainer::instance()->
                get_texture_2d_manager()->create_object("entry points");
            GLTexture2DPtr exit_tex = GLResourceManagerContainer::instance()->
                get_texture_2d_manager()->create_object("exit points");

            _res_shield.add_shield<GLTexture2D>(entry_tex);
            _res_shield.add_shield<GLTexture2D>(exit_tex);

            _entry_points_texture.reset(new GPUCanvasPair(entry_tex));
            _exit_points_texture.reset(new GPUCanvasPair(exit_tex));
        } else {
            CudaSurface2DPtr entry_surface = CudaResourceManager::instance()->create_cuda_surface_2d("entry points");
            CudaSurface2DPtr exit_surface = CudaResourceManager::instance()->create_cuda_surface_2d("exit points");

            _entry_points_texture.reset(new GPUCanvasPair(entry_surface));
            _exit_points_texture.reset(new GPUCanvasPair(exit_surface));
        }
    }
}

void EntryExitPoints::initialize() {
    if (_has_init) {
        return;
    }

    if (GL_BASE == _gpu_platform) {
        CHECK_GL_ERROR;

        //TODO change GL_RGBA32F->GL_RGBA16

        _entry_points_texture->get_gl_resource()->initialize();
        _exit_points_texture->get_gl_resource()->initialize();

        _entry_points_texture->get_gl_resource()->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _entry_points_texture->get_gl_resource()->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT, NULL);
        _entry_points_texture->get_gl_resource()->unbind();

        _exit_points_texture->get_gl_resource()->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _exit_points_texture->get_gl_resource()->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT, NULL);
        _exit_points_texture->get_gl_resource()->unbind();

        CHECK_GL_ERROR;
    } else {
        _entry_points_texture->get_cuda_resource()->load(32, 32, 32, 32, cudaChannelFormatKindFloat, _width, _height, nullptr);
        _exit_points_texture->get_cuda_resource()->load(32, 32, 32, 32, cudaChannelFormatKindFloat, _width, _height, nullptr);
    }
    _has_init = true;

}

EntryExitPoints::~EntryExitPoints() {}

GPUPlatform EntryExitPoints::get_gpu_platform() const {
    return _gpu_platform;
}

void EntryExitPoints::set_display_size(int width, int height) {
    if (_width == width && _height == height) {
        return;
    }

    _width = width;
    _height = height;

    //------------------------------------------------------//
    // when GL texture change size(call glTexSubImage*D), CUDA's interoperated resouece will invalid. it will cause:
    // 1. catch GL's errorr: GL_INVALID_OPERATION, when call glTexSubImage*D
    // 2. catch CUDA's error : invalid resource handle(33), when access interoperated texture memory.
    // solution: unregister CUDA's interoperated resource before call glTexSubImage, and re-register after call glTexSubImage
    //------------------------------------------------------//
    if (_resize_callback) {
        _resize_callback->execute(width, height);
    }

    if (CPU_BASE == _strategy) {
        _entry_points_buffer.reset(new Vector4f[_width * _height]);
        _exit_points_buffer.reset(new Vector4f[_width * _height]);
    }    

    if (GPU_BASE == _strategy && _has_init) {
        if (GL_BASE == _gpu_platform) {

            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_2D, _entry_points_texture->get_gl_resource(), GL_CLAMP_TO_BORDER, GL_LINEAR,
                GL_RGBA32F, _width, _height, 0, GL_RGBA, GL_FLOAT, nullptr);

            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_2D, _exit_points_texture->get_gl_resource(), GL_CLAMP_TO_BORDER, GL_LINEAR,
                GL_RGBA32F, _width, _height, 0, GL_RGBA, GL_FLOAT, nullptr);
        } else {
            _entry_points_texture->get_cuda_resource()->load(32, 32, 32, 32, cudaChannelFormatKindFloat, _width, _height, nullptr);
            _exit_points_texture->get_cuda_resource()->load(32, 32, 32, 32, cudaChannelFormatKindFloat, _width, _height, nullptr);
        }
    }
}

void EntryExitPoints::get_display_size(int& width, int& height) {
    width = _width;
    height = _height;
}

GPUCanvasPairPtr EntryExitPoints::get_entry_points_texture() {
    return _entry_points_texture;
}

GPUCanvasPairPtr EntryExitPoints::get_exit_points_texture() {
    return _exit_points_texture;
}

Vector4f* EntryExitPoints::get_entry_points_array() {
    return _entry_points_buffer.get();
}

Vector4f* EntryExitPoints::get_exit_points_array() {
    return _exit_points_buffer.get();
}

void EntryExitPoints::set_volume_data(std::shared_ptr<ImageData> image_data) {
    _volume_data = image_data;
}

std::shared_ptr<ImageData> EntryExitPoints::get_volume_data() const {
    return _volume_data;
}

void EntryExitPoints::set_camera(std::shared_ptr<CameraBase> camera) {
    _camera = camera;
}

std::shared_ptr<CameraBase> EntryExitPoints::get_camera() const {
    return _camera;
}

void EntryExitPoints::set_camera_calculator(
    std::shared_ptr<CameraCalculator> camera_cal) {
    _camera_calculator = camera_cal;
}

std::shared_ptr<CameraCalculator> EntryExitPoints::get_camera_calculator() const {
    return _camera_calculator;
}

void EntryExitPoints::register_resize_callback(std::shared_ptr<IEntryExitPointsResizeCallback> cb) {
    _resize_callback = cb;
}

namespace {
    void rgba32f_tp_rgb8(float* rgba32f, unsigned char* rgb8, unsigned int length, unsigned int* dim) {
        float dim_r[3] = { 1.0f / (float)dim[0], 1.0f / (float)dim[1], 1.0f / (float)dim[2] };
        unsigned char r, g, b;
        float rr, gg, bb;
        for (unsigned int i = 0; i < length; ++i) {
            rr = rgba32f[i*4] * dim_r[0] * 255.0f;
            gg = rgba32f[i*4+1] * dim_r[1] * 255.0f;
            bb = rgba32f[i*4+2] * dim_r[2] * 255.0f;

            rr = rr > 255.0f ? 255.0f : rr;
            rr = rr < 0.0f ? 0.0f : rr;

            gg = gg > 255.0f ? 255.0f : gg;
            gg = gg < 0.0f ? 0.0f : gg;

            bb = bb > 255.0f ? 255.0f : bb;
            bb = bb < 0.0f ? 0.0f : bb;

            r = (unsigned char)rr;
            g = (unsigned char)gg;
            b = (unsigned char)bb;

            rgb8[i * 3] = r;
            rgb8[i * 3 + 1] = g;
            rgb8[i * 3 + 2] = b;
        }
    }
}
void EntryExitPoints::debug_output_entry_points(const std::string& file_name) {
    if (nullptr == _volume_data) {
        MI_RENDERALGO_LOG(MI_WARNING) << "output entry points failed: volume data is null";
        return;
    }
    std::unique_ptr<unsigned char[]> rgb8(new unsigned char[_width * _height * 3]);
    if (CPU_BASE == _strategy) {
        rgba32f_tp_rgb8((float*)(_entry_points_buffer.get()), rgb8.get(), _width*_height, _volume_data->_dim);
        
    } else {
        std::unique_ptr<float[]> rgbaf4(new float[_width*_height * 4]);
        if (GL_BASE == _gpu_platform) {
            CHECK_GL_ERROR;
            GLUtils::set_pixel_unpack_alignment(1);
            _entry_points_texture->get_gl_resource()->bind();
            _entry_points_texture->get_gl_resource()->download(GL_RGBA, GL_FLOAT, rgbaf4.get());
            _entry_points_texture->get_gl_resource()->unbind();
            CHECK_GL_ERROR;
        } else { 
            if (0 != _entry_points_texture->get_cuda_resource()->download(_width * _height * 4 * sizeof(float), rgbaf4.get())) {
                MI_RENDERALGO_LOG(MI_WARNING) << "download entry points failed.";
                return;
            }
        }
        rgba32f_tp_rgb8(rgbaf4.get(), rgb8.get(), _width*_height, _volume_data->_dim);
    }
    FileUtil::write_raw(file_name, rgb8.get(), _width * _height * 3);
}

void EntryExitPoints::debug_output_exit_points(const std::string& file_name) {
    if (nullptr == _volume_data) {
        MI_RENDERALGO_LOG(MI_WARNING) << "output exit points failed: volume data is null";
        return;
    }
    std::unique_ptr<unsigned char[]> rgb8(new unsigned char[_width * _height * 3]);
    if (CPU_BASE == _strategy) {
        rgba32f_tp_rgb8((float*)(_exit_points_buffer.get()), rgb8.get(), _width*_height, _volume_data->_dim);

    }
    else {
        std::unique_ptr<float[]> rgbaf4(new float[_width*_height * 4]);
        if (GL_BASE == _gpu_platform) {
            CHECK_GL_ERROR;
            GLUtils::set_pixel_unpack_alignment(1);
            _exit_points_texture->get_gl_resource()->bind();
            _exit_points_texture->get_gl_resource()->download(GL_RGBA, GL_FLOAT, rgbaf4.get());
            _exit_points_texture->get_gl_resource()->unbind();
            CHECK_GL_ERROR;
        }
        else {
            if (0 != _exit_points_texture->get_cuda_resource()->download(_width * _height * 4 * sizeof(float), rgbaf4.get())) {
                MI_RENDERALGO_LOG(MI_WARNING) << "download exit points failed.";
                return;
            }
        }
        rgba32f_tp_rgb8(rgbaf4.get(), rgb8.get(), _width*_height, _volume_data->_dim);
    }
    FileUtil::write_raw(file_name, rgb8.get(), _width * _height * 3);
}

MED_IMG_END_NAMESPACE