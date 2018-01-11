#include "mi_cuda_surface_2d.h"
#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaSurface2D::CudaSurface2D(UIDType uid) : CudaSurfaceBase(uid, "CudaSurface2D"), _width(0), _height(0) {
}

CudaSurface2D::~CudaSurface2D() {
    finalize();
}

void CudaSurface2D::finalize() {
    CudaSurfaceBase::finalize();
    _width = 0;
    _height = 0;
}

float CudaSurface2D::memory_used() const {
    return _width*_height*CudaUtils::get_component_byte(_channel) / 1024.0f;
}

int CudaSurface2D::get_width() const {
    return _width;
}

int CudaSurface2D::get_height() const {
    return _height;
}

int CudaSurface2D::load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, void* h_data) {
    if (width <= 0 || height <= 0) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "load invalid size " << width << " " << height << " to texture 2D.";
        return -1;
    }

    if (_d_array) {
        if (_channel[0] != channel_x || _channel[1] != channel_y ||
            _channel[2] != channel_z || _channel[3] != channel_w ||
            width != _width || height != _height) {
            //re-create
            this->finalize();
        }
    }

    if (nullptr == _d_array) {
        _channel[0] = channel_x;
        _channel[1] = channel_y;
        _channel[2] = channel_z;
        _channel[3] = channel_w;
        _format = format;
        _width = width;
        _height = height;
        cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(channel_x, channel_y, channel_z, channel_w, format);
        cudaError_t err = cudaMallocArray(&_d_array, &format_desc, width, height, cudaArraySurfaceLoadStore);
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

int CudaSurface2D::update(int offset_x, int offset_y, int width, int height, void* h_data) {
    cudaError_t err = cudaMemcpyToArray(_d_array, offset_x, offset_y, h_data, width*height*CudaUtils::get_component_byte(_channel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

int CudaSurface2D::download(unsigned int size, void* h_data) {
    const unsigned int cur_size = (unsigned int)_width*(unsigned int)_height*(unsigned int)CudaUtils::get_component_byte(_channel);
    if (size != cur_size) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "invalid size: " << size << " when download cuda 2D surface with size: " << cur_size;
        return -1;
    }
    cudaError_t err = cudaMemcpyFromArray(h_data, _d_array, 0, 0, cur_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

MED_IMG_END_NAMESPACE