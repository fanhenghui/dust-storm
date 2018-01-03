#include "mi_cuda_texture_3d.h"
#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaTexture3D::CudaTexture3D(UIDType uid) : CUDATextureBase(uid, "CudaTexture3D") {
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
    _format = cudaChannelFormatKindNone;
    _width = 0;
    _height = 0;
    _depth = 0;
}

CudaTexture3D::~CudaTexture3D() {

}

void CudaTexture3D::finalize() {

}

int CudaTexture3D::load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, int depth, void* data) {
    //malloc and load, or update all
    if (_cuda_array) {
        if (_channel[0] != channel_x || _channel[1] != channel_y ||
            _channel[2] != channel_z || _channel[3] != channel_w || 
            _width != width || _height != height || depth != _depth) {
            MI_CUDARESOURCE_LOG(MI_ERROR) << "load different format array to CUDA texture 1D. init foramt {ch:"
                << _channel[0] << " " << _channel[1] << " " << _channel[2] << " " << _channel[3] << ", format: " << _format << ", extent: " << _width << " " << _height << " " << _depth
                << "}. call load func foramt {ch: "
                << channel_x << " " << channel_y << " " << channel_z << " " << channel_w << ", format: " << format << ", extent: " << width << " " << height << " " << depth << "}.";
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
        _depth = depth;

        cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(channel_x, channel_y, channel_z, channel_w, format);
        const cudaExtent extent = { width, height, depth };
        cudaError_t err = cudaMalloc3DArray(&_cuda_array, &format_desc, extent, NULL);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            return -1;
        }
    }
    
    const cudaExtent extent = { width, height, depth };
    cudaMemcpy3DParms memcpy_3d_parms;
    memset(&memcpy_3d_parms, 0, sizeof(cudaMemcpy3DParms));
    memcpy_3d_parms.srcPtr = make_cudaPitchedPtr(data, _width*CudaUtils::get_componet_byte(_channel), _width, _height);
    memcpy_3d_parms.extent = extent;
    memcpy_3d_parms.kind = cudaMemcpyHostToDevice;
    memcpy_3d_parms.dstArray = _cuda_array;
    cudaError_t err = cudaMemcpy3D(&memcpy_3d_parms);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

int CudaTexture3D::update(int offset_x, int offset_y, int offset_z, int width, int height, int depth, void* data) {
    if (nullptr == _cuda_array) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << *this << " try update texture 3D with null CUDA array.";
        return -1;
    }

    cudaExtent extent = { width, height, depth };
    cudaMemcpy3DParms memcpy_3d_parms;
    memset(&memcpy_3d_parms, 0, sizeof(cudaMemcpy3DParms));
    memcpy_3d_parms.srcPtr = make_cudaPitchedPtr(data, width*CudaUtils::get_componet_byte(_channel), width, height);
    memcpy_3d_parms.extent = extent;
    memcpy_3d_parms.kind = cudaMemcpyHostToDevice;
    memcpy_3d_parms.dstPos.x = offset_x;
    memcpy_3d_parms.dstPos.y = offset_y;
    memcpy_3d_parms.dstPos.z = offset_z;
    memcpy_3d_parms.dstArray = _cuda_array;
    cudaError_t err = cudaMemcpy3D(&memcpy_3d_parms);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

cudaTextureObject_t CudaTexture3D::create_object(
    cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode, bool normalized_coords) {

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = _cuda_array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.addressMode[2] = address_mode;
    tex_desc.filterMode = filter_mode;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = normalized_coords;

    cudaTextureObject_t tex_obj(0);
    cudaError_t err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
    CHECK_CUDA_ERROR(err);

    return tex_obj;
}

MED_IMG_END_NAMESPACE