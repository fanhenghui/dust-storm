#include "mi_gpu_image_compressor.h"

#include <cuda_runtime.h>

#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"

#include "glresource/mi_gl_texture_2d.h"
#include "cudaresource/mi_cuda_surface_2d.h"
#include "renderalgo/mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

extern "C"
cudaError_t surface_2d_rgba8_flip_vertical_to_global_memory_rgb8(cudaSurfaceObject_t sur_rgba8, int width, int height, unsigned char* d_rgb_8);

struct GPUImgCompressor::InnerParams{
    // encoder
    gpujpeg_encoder* encoder;             
    // encoding input
    gpujpeg_encoder_input encoder_input;  
    // gpujpeg's texture (GL_BASE)
    gpujpeg_opengl_texture* gpujpeg_texture;

    InnerParams():encoder(nullptr), gpujpeg_texture(nullptr) {

    }

    void release() {
        if (nullptr != gpujpeg_texture) {
            gpujpeg_opengl_texture_unregister(gpujpeg_texture);
            gpujpeg_texture = nullptr;
        }
        if (nullptr != encoder) {
            gpujpeg_encoder_destroy(encoder);
            encoder = nullptr;
        }
    }
};

GPUImgCompressor::GPUImgCompressor(GPUPlatform platform): _gpu_platform(platform), _duration(0.0f) {
}

GPUImgCompressor::~GPUImgCompressor() {

}

int GPUImgCompressor::set_image(GPUCanvasPairPtr canvas, const std::vector<int>& qualitys) {

    if (nullptr == canvas) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input null GPU canvas pair to GPU compressor.";
        return -1;
    }
    
    //check input quality
    for (auto it = qualitys.begin(); it != qualitys.end(); ++it) {
        if (*it < 1 || *it > 100) {
            MI_RENDERALGO_LOG(MI_ERROR) << "invalid input quality: " << *it << " to GPU compressor";
            return -1;
        }
    }

    //check input platform
    if (GL_BASE == _gpu_platform && !canvas->gl()) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input invalid gl based canvas to GPU compressor.";
        return -1;
    } else if (CUDA_BASE == _gpu_platform && !canvas->cuda()) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input invalid cuda based canvas to GPU compressor.";
        return -1;
    }

    //release
    for (auto it = _params.begin(); it != _params.end(); ++it) {
        it->second.release();
    }
    _params.clear();

    //create
    _canvas = canvas;
    if (GL_BASE == _gpu_platform) {
        GLTexture2DPtr tex = canvas->get_gl_resource();
        if (nullptr == tex) {
            MI_RENDERALGO_LOG(MI_ERROR) << "input invalid null texture to GPU compressor.";
            return -1;
        }

        const int width = tex->get_width();
        const int height = tex->get_height();
        if (tex->get_format() != GL_RGB) {
            MI_RENDERALGO_LOG(MI_ERROR) << "GPU compressor just support GL_RGB texture.";
            return -1;
        }
        if (tex->get_id() == 0) {
            MI_RENDERALGO_LOG(MI_ERROR) << "input invalid 0 texture to GPU compressor.";
            return -1;
        }

        for (auto it = qualitys.begin(); it != qualitys.end(); ++it) {
            gpujpeg_parameters params;
            gpujpeg_set_default_parameters(&params);        //default parameter
            gpujpeg_parameters_chroma_subsampling(&params); //default sampling parameter
            params.quality = *it;

            gpujpeg_image_parameters image_param;
            gpujpeg_image_set_default_parameters(&image_param);
            image_param.width = width;
            image_param.height = height;
            image_param.comp_count = 3;
            image_param.color_space = GPUJPEG_RGB;
            image_param.sampling_factor = GPUJPEG_4_4_4;

            //create gpujpeg's texture to bind GL texture
            gpujpeg_opengl_texture* gpujpeg_texture = gpujpeg_opengl_texture_register(tex->get_id(), GPUJPEG_OPENGL_TEXTURE_READ);
            // create encoder
            gpujpeg_encoder* encoder = gpujpeg_encoder_create(&params, &image_param);
            if (nullptr == encoder) {
                MI_RENDERALGO_LOG(MI_ERROR) << "create GPU compressor failed.";
                return -1;
            }
            // set texture as input
            struct gpujpeg_encoder_input encoder_input;
            gpujpeg_encoder_input_set_texture(&encoder_input, gpujpeg_texture);

            InnerParams inner_params;
            inner_params.encoder = encoder;
            inner_params.encoder_input = encoder_input;
            inner_params.gpujpeg_texture = gpujpeg_texture;

            _params[*it] = inner_params;
        }
        
    } else {
        CudaSurface2DPtr surface = canvas->get_cuda_resource();
        if (nullptr == surface) {
            MI_RENDERALGO_LOG(MI_ERROR) << "input invalid null surface to GPU compressor.";
            return -1;
        }

        const int width = surface->get_width();
        const int height = surface->get_height();
        int channel[4] = {0};
        surface->get_channel(channel);
        if (!(channel[0] == 8 && channel[1] == 8 && channel[2] == 8 && channel[3] == 8)) {
            MI_RENDERALGO_LOG(MI_ERROR) << "GPU compressor just support channel [8,8,8,8], input channel is [ " <<
                channel[0] << " " << channel[1] << " " << channel[2] << " " << channel[3] << "].";
            return -1;
        }
        if (surface->get_object() == 0) {
            MI_RENDERALGO_LOG(MI_ERROR) << "input invalid 0 surface object to GPU compressor.";
            return -1;
        }

        for (auto it = qualitys.begin(); it != qualitys.end(); ++it) {
            gpujpeg_parameters params;
            gpujpeg_set_default_parameters(&params);        //default parameter
            gpujpeg_parameters_chroma_subsampling(&params); //default sampling parameter
            params.quality = *it;

            gpujpeg_image_parameters image_param;
            gpujpeg_image_set_default_parameters(&image_param);
            image_param.width = width;
            image_param.height = height;
            image_param.comp_count = 3;
            image_param.color_space = GPUJPEG_RGB;
            image_param.sampling_factor = GPUJPEG_4_4_4;

            // create encoder
            gpujpeg_encoder* encoder = gpujpeg_encoder_create(&params, &image_param);
            if (nullptr == encoder) {
                MI_RENDERALGO_LOG(MI_ERROR) << "create GPU compressor failed.";
                return -1;
            }

            // set empty input
            gpujpeg_encoder_input encoder_input;
            gpujpeg_image_destroy(encoder_input.image);
            encoder_input.type = GPUJPEG_ENCODER_INPUT_INTERNAL_BUFFER;

            InnerParams inner_params;
            inner_params.encoder = encoder;
            inner_params.encoder_input = encoder_input;

            _params[*it] = inner_params;
        }
    }
    return 0;
}

int GPUImgCompressor::resize_image(int width, int height) {
    if (_params.empty()) {
        return 0;
    }

    std::vector<int> qualitys;
    for (auto it = _params.begin(); it != _params.end(); ++it) {
        qualitys.push_back(it->first);
    }

    return this->set_image(_canvas, qualitys);
}

float GPUImgCompressor::get_last_duration() const {
    return _duration;
}

int GPUImgCompressor::compress(int quality, void* buffer, int& compress_size) {
    auto it = _params.find(quality);
    if (it == _params.end()) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input not existed quality: " << quality << " to GPU compressor.";
        return -1;
    }
    if (nullptr == buffer) {
        MI_RENDERALGO_LOG(MI_ERROR) << "input null buffer to GPU compressor.";
        return -1;
    }

    InnerParams& params = it->second;
    if (GL_BASE == _gpu_platform) {
        //---------------------------------------------------//
        // GL_BASE: use GL texture as input, compress directly
        //---------------------------------------------------//
        uint8_t* image_compressed = nullptr;
        int err = gpujpeg_encoder_encode(params.encoder, &params.encoder_input,
            &image_compressed, &compress_size);
        if (err != 0) {
            MI_RENDERALGO_LOG(MI_ERROR) << "GPU compress failed.";
        }
        memcpy(buffer, image_compressed, compress_size);

    } else {
        //-------------------------------------------------------------------------//
        // CUDA_BASE: kernel write RGBA8 surface to GPUJPEG's coder's inner raw_data
        //-------------------------------------------------------------------------//
        CudaSurface2DPtr surface = _canvas->get_cuda_resource();
        unsigned char* d_rgb = (unsigned char*)gpujpeg_encoder_get_inner_device_image_data(params.encoder);
        cudaError_t err = surface_2d_rgba8_flip_vertical_to_global_memory_rgb8(surface->get_object(),
            surface->get_width(), surface->get_height(), d_rgb);
        if (err != cudaSuccess) {

        }
    }

    return 0;
}

MED_IMG_END_NAMESPACE
