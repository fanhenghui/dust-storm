#include "mi_cuda_texture_1d.h"
#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaTexture1D::CudaTexture1D(UIDType uid): CUDATextureBase(uid, "CudaTexture1D") {
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
    _format = cudaChannelFormatKindNone;
    _length = 0;
}

CudaTexture1D::~CudaTexture1D() {

}

void CudaTexture1D::finalize() {
    
}

int CudaTexture1D::load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int length, void* data) {
    //malloc and load, or update all
    if (_cuda_array) {
        if (_channel[0] != channel_x || _channel[1] != channel_y ||
            _channel[2] != channel_z || _channel[3] != channel_w || length != _length) {
            MI_CUDARESOURCE_LOG(MI_ERROR) << "load different format array to CUDA texture 1D. init foramt {ch:"
                << _channel[0] << " " << _channel[1] << " " << _channel[2] << " " << _channel[3] << ", format: " << _format << ", length: " << _length 
                << "}. call load func foramt {ch: "
                << channel_x << " " << channel_y << " " << channel_z << " " << channel_w << ", format: " << format << ", length: " << length << "}.";
            return -1;
        }
    } else {
        _channel[0] = channel_x;
        _channel[1] = channel_y;
        _channel[2] = channel_z;
        _channel[3] = channel_w;
        _format = _format;
        _length = length;
        cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(channel_x, channel_y, channel_z, channel_w, format);
        cudaError_t err = cudaMallocArray(&_cuda_array, &format_desc, length, 1);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            return -1;
        }
    }

    cudaError_t err = cudaMemcpyToArray(_cuda_array, 0, 0, data, length*CudaUtils::get_componet_byte(_channel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

int CudaTexture1D::update(int offset, int length, void* data) {
    cudaError_t err = cudaMemcpyToArray(_cuda_array, offset, 0, data, length*CudaUtils::get_componet_byte(_channel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

cudaTextureObject_t CudaTexture1D::create_object(
    cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode, bool normalized_coords) {

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = _cuda_array;
    
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = address_mode;
    tex_desc.filterMode = filter_mode;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = normalized_coords;

    cudaTextureObject_t tex_obj(0);
    cudaError_t err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
    CHECK_CUDA_ERROR(err);

    return tex_obj;
}

MED_IMG_END_NAMESPACE