#include "mi_cuda_texture_2d.h"
#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaTexture2D::CudaTexture2D(UIDType uid) : CudaTextureBase(uid, "CudaTexture2D"), _width(0), _height(0){

}

CudaTexture2D::~CudaTexture2D() {

}

float CudaTexture2D::memory_used() const {
    return _width*_height*CudaUtils::get_component_byte(_channel) / 1024.0f;
}


int CudaTexture2D::get_width() const {
    return _width;
}

int CudaTexture2D::get_height() const {
    return _height;
}

int CudaTexture2D::load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, void* h_data) {
    if (width <= 0 || height <=0) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "load invalid size " << width << " " << height << " to texture 2D.";
        return -1;
    }

    //malloc and load, or update all
    if (_d_array) {
        if (_channel[0] != channel_x || _channel[1] != channel_y ||
            _channel[2] != channel_z || _channel[3] != channel_w || 
            width != _width || height != _height) {
            MI_CUDARESOURCE_LOG(MI_ERROR) << "load different format array to CUDA texture 1D. init foramt {ch:"
                << _channel[0] << " " << _channel[1] << " " << _channel[2] << " " << _channel[3] << ", format: " << _format << ", size: " << _width << " " << _height
                << "}. call load func foramt {ch: "
                << channel_x << " " << channel_y << " " << channel_z << " " << channel_w << ", format: " << format << ", size: " << width << " " << height << "}.";
            return -1;
        }
    }
    else {
        _channel[0] = channel_x;
        _channel[1] = channel_y;
        _channel[2] = channel_z;
        _channel[3] = channel_w;
        _format = _format;
        _width = width;
        _height = height;
        cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(channel_x, channel_y, channel_z, channel_w, format);
        cudaError_t err = cudaMallocArray(&_d_array, &format_desc, width, height);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            return -1;
        }
    }

    if (nullptr == h_data) {
        return 0;
    }

    cudaError_t err = cudaMemcpyToArray(_d_array, 0, 0, h_data, width*height*CudaUtils::get_component_byte(_channel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

int CudaTexture2D::update(int offset_x, int offset_y, int width, int height, void* h_data) {
    cudaError_t err = cudaMemcpyToArray(_d_array, offset_x, offset_y, h_data, width*height*CudaUtils::get_component_byte(_channel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

int CudaTexture2D::download(unsigned int size, void* h_data) {
    const unsigned int cur_size = (unsigned int)_width*(unsigned int)_height*(unsigned int)CudaUtils::get_component_byte(_channel);
    if (size != cur_size) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "invalid size: " << size << " when download cuda 2D texture with size: " << cur_size;
        return -1;
    }
    cudaError_t err = cudaMemcpyFromArray(h_data, _d_array, 0, 0, cur_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "download cuda 2D texture failed: " << err;
        return -1;
    }
    else {
        return 0;
    }
}

cudaTextureObject_t CudaTexture2D::create_object(
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

MED_IMG_END_NAMESPACE