#include "mi_cuda_gl_texture_2d.h"

#include "GL/glew.h"

#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"
#include "glresource/mi_gl_texture_2d.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
    inline int fill_cuda_texture_parameter(GLenum format, GLenum type, int(&channel)[4]) {
        switch (format) {
        case GL_RED: {
            if (type = GL_UNSIGNED_BYTE || type == GL_BYTE) {
                channel[0] = 8;
                channel[1] = 0;
                channel[2] = 0;
                channel[3] = 0;
            }
            else if (type == GL_UNSIGNED_SHORT || type == GL_SHORT) {
                channel[0] = 16;
                channel[1] = 0;
                channel[2] = 0;
                channel[3] = 0;
            }
            else if (type == GL_FLOAT) {
                channel[0] = 32;
                channel[1] = 0;
                channel[2] = 0;
                channel[3] = 0;
            }
            else {
                MI_CUDARESOURCE_LOG(MI_ERROR) << "cuda gl texture RGB can't support type: " << type;
                return -1;
            }
            break;
        }
        case GL_RGB: {
            if (type = GL_UNSIGNED_BYTE || type == GL_BYTE) {
                MI_CUDARESOURCE_LOG(MI_ERROR) << "cuda gl texture can't support GL_RGB8 format.";
                return -1;
            }
            else if (type == GL_UNSIGNED_SHORT || type == GL_SHORT) {
                channel[0] = 16;
                channel[1] = 16;
                channel[2] = 16;
                channel[3] = 0;
            }
            else if (type == GL_FLOAT) {
                channel[0] = 32;
                channel[1] = 32;
                channel[2] = 32;
                channel[3] = 0;
            }
            else {
                MI_CUDARESOURCE_LOG(MI_ERROR) << "cuda gl texture RGB can't support type: " << type;
                return -1;
            }
            break;
        }
        case GL_RGBA: {
            if (type = GL_UNSIGNED_BYTE || type == GL_BYTE) {
                channel[0] = 8;
                channel[1] = 8;
                channel[2] = 8;
                channel[3] = 8;
            }
            else if (type == GL_UNSIGNED_SHORT || type == GL_SHORT) {
                channel[0] = 16;
                channel[1] = 16;
                channel[2] = 16;
                channel[3] = 16;
            }
            else if (type == GL_FLOAT) {
                channel[0] = 32;
                channel[1] = 32;
                channel[2] = 32;
                channel[3] = 32;
            }
            else {
                MI_CUDARESOURCE_LOG(MI_ERROR) << "cuda gl texture RGB can't support type: " << type;
                return -1;
            }
            break;
        }
        default:
            MI_CUDARESOURCE_LOG(MI_ERROR) << "cuda gl texture can't support format: " << format;
            return -1;
            break;
        }
        return 0;
    }
}

CudaGLTexture2D::CudaGLTexture2D(UIDType uid) : CudaTextureBase(uid, "CudaGLTexture2D"), _cuda_graphic_resource(nullptr){
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
    _format = cudaChannelFormatKindNone;
    _width = 0;
    _height = 0;
}

CudaGLTexture2D::~CudaGLTexture2D() {

}

void CudaGLTexture2D::finalize() {
    if (_cuda_graphic_resource) {
        cudaError_t err = cudaGraphicsUnregisterResource(_cuda_graphic_resource);
        _cuda_graphic_resource = nullptr;
        CHECK_CUDA_ERROR(err);
    }
}

float CudaGLTexture2D::memory_used() const {
    return _width*_height*CudaUtils::get_component_byte(_channel) / 1024.0f;
}

cudaTextureObject_t CudaGLTexture2D::create_object(
    cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode, bool normalized_coords) {

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = _d_array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.filterMode = filter_mode;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = normalized_coords;

    cudaTextureObject_t tex_obj(0);
    cudaError_t err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
    CHECK_CUDA_ERROR(err);

    return tex_obj;
}

int CudaGLTexture2D::register_gl_texture(std::shared_ptr<GLTexture2D> tex, cudaGraphicsRegisterFlags register_flag) {
    if (register_flag != cudaGraphicsRegisterFlagsNone &&
        register_flag != cudaGraphicsRegisterFlagsWriteDiscard &&
        register_flag != cudaGraphicsRegisterFlagsReadOnly) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "invalid unsupported cudaGraphicsRegisterFlags: " << register_flag;
        return -1;
    }
    _register_flag = register_flag;
    _width = tex->get_width();
    _height = tex->get_height();
    if (0 != fill_cuda_texture_parameter(tex->get_format(), tex->get_data_type(), _channel)) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "invalid gl texture when cuda register gl texture ";
        return -1;
    }

    if (_cuda_graphic_resource) {
        cudaError_t err = cudaGraphicsUnregisterResource(_cuda_graphic_resource);
        _cuda_graphic_resource = nullptr;
        CHECK_CUDA_ERROR(err);
    }

    tex->bind();
    cudaError_t err = cudaGraphicsGLRegisterImage(&_cuda_graphic_resource, tex->get_id(), GL_TEXTURE_2D, register_flag);
    tex->unbind();
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1; 
    } else {
        return 0;
    }
}

int CudaGLTexture2D::map_gl_texture() {
    cudaError_t err = cudaGraphicsMapResources(1, &_cuda_graphic_resource);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    } 

    err = cudaGraphicsSubResourceGetMappedArray(&_d_array, _cuda_graphic_resource, 0,0);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    } else {
        return 0;
    }
}

int CudaGLTexture2D::unmap_gl_texture() {
    cudaError_t err = cudaGraphicsUnmapResources(1, &_cuda_graphic_resource);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    } else {
        return 0;
    }
}

int CudaGLTexture2D::write_to_gl_texture(void* array, size_t count, cudaMemcpyKind memcpy_kind) {
    if (_register_flag == cudaGraphicsRegisterFlagsReadOnly) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << *this << " can't write to read only texture.";
        return -1;
    }

    cudaError_t err = cudaMemcpyToArray(_d_array, 0, 0, array, count, memcpy_kind);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    } else {
        return 0;
    }
}

int CudaGLTexture2D::download(unsigned int size, void* h_data) {
    const unsigned int cur_size = (unsigned int)_width*(unsigned int)_height*(unsigned int)CudaUtils::get_component_byte(_channel);
    if (size != cur_size) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "invalid size: " << size << " when download cuda 2D texture with size: " << cur_size;
        return -1;
    }
    cudaError_t err = cudaMemcpyFromArray(h_data, _d_array, 0, 0, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "download cuda 2D texture failed: " << err;
        return -1;
    }
    else {
        return 0;
    }
}

MED_IMG_END_NAMESPACE