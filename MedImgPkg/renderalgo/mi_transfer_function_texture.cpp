#include "mi_transfer_function_texture.h"

#include "util/mi_memory_shield.h"

#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_texture_1d_array.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_texture_cache.h"

#include "cudaresource/mi_cuda_texture_1d.h"
#include "cudaresource/mi_cuda_texture_1d_array.h"
#include "cudaresource/mi_cuda_resource_manager.h"

#include "mi_color_transfer_function.h"
#include "mi_opacity_transfer_function.h"

MED_IMG_BEGIN_NAMESPACE

TransferFunctionTexture::TransferFunctionTexture(RayCastingStrategy strategy, GPUPlatform platform)
:_strategy(strategy), _gpu_platform(platform), _label_level(L_8), _init(false){

}

void TransferFunctionTexture::initialize(LabelLevel label_level) {
    if (_init && _label_level == label_level) {
        return;
    }
    _label_level = label_level;

    if (GPU_BASE == _strategy) {
        // initialize gray pseudo color texture
        if (GL_BASE == _gpu_platform) {
            if (!_pseudo_color_texture) {
                GLTexture1DPtr pseudo_color_texture = GLResourceManagerContainer::instance()->
                    get_texture_1d_manager()->create_object("pseudo color");
                _pseudo_color_texture.reset(new GPUTexture1DPair(pseudo_color_texture));
                _res_shield.add_shield<GLTexture1D>(pseudo_color_texture);

                unsigned char* gray_array = new unsigned char[S_TRANSFER_FUNC_WIDTH * 3];
                for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
                    gray_array[i * 3] = static_cast<unsigned char>(255.0f * (float)i / (float)S_TRANSFER_FUNC_WIDTH);
                    gray_array[i * 3 + 1] = gray_array[i * 3];
                    gray_array[i * 3 + 2] = gray_array[i * 3];
                }

                GLTextureCache::instance()->cache_load(
                    GL_TEXTURE_1D, pseudo_color_texture, GL_CLAMP_TO_EDGE, GL_LINEAR,
                    GL_RGB8, S_TRANSFER_FUNC_WIDTH, 0, 0, GL_RGB, GL_UNSIGNED_BYTE, (char*)gray_array);
            }

            //release old color opacity texture array
            if (_color_opacity_texture_array) {
                _res_shield.remove_shield(_color_opacity_texture_array->get_gl_resource());
            }

            GLTexture1DArrayPtr color_opacity_texture_array = GLResourceManagerContainer::instance()->
                get_texture_1d_array_manager()->create_object("color opacity texture array");
            _color_opacity_texture_array.reset(new GPUTexture1DArrayPair(color_opacity_texture_array));
            _res_shield.add_shield<GLTexture1DArray>(color_opacity_texture_array);

            const int tex_num = static_cast<int>(label_level);
            unsigned char* rgba = new unsigned char[S_TRANSFER_FUNC_WIDTH * tex_num * 4];
            memset(rgba, 0, S_TRANSFER_FUNC_WIDTH * tex_num * 4);

            GLTextureCache::instance()->cache_load(
                GL_TEXTURE_1D_ARRAY, color_opacity_texture_array, GL_CLAMP_TO_EDGE,
                GL_LINEAR, GL_RGBA8, S_TRANSFER_FUNC_WIDTH, tex_num, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, (char*)rgba);
        }
        else {
            if (!_pseudo_color_texture) {
                CudaTexture1DPtr pseudo_color_texture = CudaResourceManager::instance()->
                    create_cuda_texture_1d("pseudo color");
                _pseudo_color_texture.reset(new GPUTexture1DPair(pseudo_color_texture));

                std::unique_ptr<unsigned char[]> gray_array(new unsigned char[S_TRANSFER_FUNC_WIDTH * 4]);
                for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
                    gray_array[i * 4] = static_cast<unsigned char>(255.0f * (float)i / (float)S_TRANSFER_FUNC_WIDTH);
                    gray_array[i * 4 + 1] = gray_array[i * 4];
                    gray_array[i * 4 + 2] = gray_array[i * 4];
                    gray_array[i * 4 + 2] = 255;
                }
                pseudo_color_texture->load(8, 8, 8, 8, cudaChannelFormatKindUnsigned, S_TRANSFER_FUNC_WIDTH, gray_array.get());
            }

            const int tex_num = static_cast<int>(label_level);
            CudaTexture1DArrayPtr color_opacity_texture_array = CudaResourceManager::instance()->
                create_cuda_texture_1d_array("color opacity texture array", tex_num);
            _color_opacity_texture_array.reset(new GPUTexture1DArrayPair(color_opacity_texture_array));
        }
    }
    else {
        // TODO CPU gray pseudo array
    }
}

TransferFunctionTexture::~TransferFunctionTexture() {

}

void TransferFunctionTexture::set_color_opacity(std::shared_ptr<ColorTransFunc> color, std::shared_ptr<OpacityTransFunc> opacity, unsigned char label) {
    if (GPU_BASE == _strategy) {
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

        RENDERALGO_CHECK_NULL_EXCEPTION(_color_opacity_texture_array);
        if (GL_BASE == _gpu_platform) {
            if (!_color_opacity_texture_array->gl()) {
                RENDERALGO_THROW_EXCEPTION("invalid color opacity transfer function texture platform");
            }
            
            GLTextureCache::instance()->cache_update(
                GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array->get_gl_resource(), 0, label, 0,
                S_TRANSFER_FUNC_WIDTH, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, (char*)rgba);
        }
        else {
            if (!_color_opacity_texture_array->cuda()) {
                RENDERALGO_THROW_EXCEPTION("invalid color opacity transfer function texture platform");
            }

            MemShield sheild((char*)rgba);
            _color_opacity_texture_array->get_cuda_resource()->load(8, 8, 8, 8, label,
                cudaChannelFormatKindUnsigned, S_TRANSFER_FUNC_WIDTH, rgba);
        }
    } else {
        RENDERALGO_THROW_EXCEPTION("CPU strategy can't support color opacity transfer function");
    }
}

GPUTexture1DArrayPairPtr TransferFunctionTexture::get_color_opacity_texture_array() {
    return _color_opacity_texture_array;
}

GPUTexture1DPairPtr TransferFunctionTexture::get_pseudo_color_texture() {
    return _pseudo_color_texture;
}

void TransferFunctionTexture::set_pseudo_color(std::shared_ptr<ColorTransFunc> color) {
    if (GPU_BASE == _strategy) {
        std::vector<ColorTFPoint> pts;
        color->set_width(S_TRANSFER_FUNC_WIDTH);
        color->get_point_list(pts);

        RENDERALGO_CHECK_NULL_EXCEPTION(_pseudo_color_texture);
        if (GL_BASE == _gpu_platform) {
            if (!_pseudo_color_texture->gl()) {
                RENDERALGO_THROW_EXCEPTION("invalid pseudo transfer function texture platform");
            }
            unsigned char* rgb = new unsigned char[S_TRANSFER_FUNC_WIDTH * 3];
            for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
                rgb[i * 3] = static_cast<unsigned char>(pts[i].x);
                rgb[i * 3 + 1] = static_cast<unsigned char>(pts[i].y);
                rgb[i * 3 + 2] = static_cast<unsigned char>(pts[i].z);
            }
            GLTextureCache::instance()->cache_update(
                GL_TEXTURE_1D, _pseudo_color_texture->get_gl_resource(), 0, 0, 0, S_TRANSFER_FUNC_WIDTH, 0,
                0, GL_RGB, GL_UNSIGNED_BYTE, (char*)rgb);
        }
        else {
            if (!_pseudo_color_texture->cuda()) {
                RENDERALGO_THROW_EXCEPTION("invalid pseudo transfer function texture platform");
            }
            std::unique_ptr<unsigned char[]> rgba(new unsigned char[S_TRANSFER_FUNC_WIDTH * 4]);
            for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
                rgba[i * 4] = static_cast<unsigned char>(pts[i].x);
                rgba[i * 4 + 1] = static_cast<unsigned char>(pts[i].y);
                rgba[i * 4 + 2] = static_cast<unsigned char>(pts[i].z);
                rgba[i * 4 + 3] = 255;
            }
            _pseudo_color_texture->get_cuda_resource()->load(8, 8, 8, 8, 
                cudaChannelFormatKindUnsigned, S_TRANSFER_FUNC_WIDTH, rgba.get());
        }
    } else {
        // TODO CPU gray pseudo array
    }
}

MED_IMG_END_NAMESPACE


