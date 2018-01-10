#include "mi_cuda_texture_3d.h"
#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaTexture3D::CudaTexture3D(UIDType uid) : CudaTextureBase(uid, "CudaTexture3D"), _width(0), _height(0), _depth(0) {
}

CudaTexture3D::~CudaTexture3D() {

}

float CudaTexture3D::memory_used() const {
    return _width*_height*_depth*CudaUtils::get_component_byte(_channel) / 1024.0f;
}

int CudaTexture3D::get_width() const {
    return _width;
}

int CudaTexture3D::get_height() const {
    return _height;
}

int CudaTexture3D::get_depth() const {
    return _depth;
}

int CudaTexture3D::load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, int depth, void* h_data) {
    if (width <= 0 || height <= 0 || depth <= 0) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "load invalid size " << width << " " << height << " " << depth << " to texture 3D.";
        return -1;
    }
    //malloc and load, or update all
    if (_d_array) {
        if (_channel[0] != channel_x || _channel[1] != channel_y ||
            _channel[2] != channel_z || _channel[3] != channel_w || 
            _width != width || _height != height || depth != _depth) {
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
        _depth = depth;

        cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(channel_x, channel_y, channel_z, channel_w, format);
        const cudaExtent extent = { width, height, depth };
        cudaError_t err = cudaMalloc3DArray(&_d_array, &format_desc, extent, NULL);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            return -1;
        }
    }

    if (nullptr == h_data) {
        return 0;
    }
    
    const cudaExtent extent = { width, height, depth };
    cudaMemcpy3DParms memcpy_3d_parms;
    memset(&memcpy_3d_parms, 0, sizeof(cudaMemcpy3DParms));
    memcpy_3d_parms.srcPtr = make_cudaPitchedPtr(h_data, _width*CudaUtils::get_component_byte(_channel), _width, _height);
    memcpy_3d_parms.extent = extent;
    memcpy_3d_parms.kind = cudaMemcpyHostToDevice;
    memcpy_3d_parms.dstArray = _d_array;
    cudaError_t err = cudaMemcpy3D(&memcpy_3d_parms);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

int CudaTexture3D::update(int offset_x, int offset_y, int offset_z, int width, int height, int depth, void* h_data) {
    if (nullptr == _d_array) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << *this << " try update texture 3D with null CUDA array.";
        return -1;
    }

    cudaExtent extent = { width, height, depth };
    cudaMemcpy3DParms memcpy_3d_parms;
    memset(&memcpy_3d_parms, 0, sizeof(cudaMemcpy3DParms));
    memcpy_3d_parms.srcPtr = make_cudaPitchedPtr(h_data, width*CudaUtils::get_component_byte(_channel), width, height);
    memcpy_3d_parms.extent = extent;
    memcpy_3d_parms.kind = cudaMemcpyHostToDevice;
    memcpy_3d_parms.dstPos.x = offset_x;
    memcpy_3d_parms.dstPos.y = offset_y;
    memcpy_3d_parms.dstPos.z = offset_z;
    memcpy_3d_parms.dstArray = _d_array;
    cudaError_t err = cudaMemcpy3D(&memcpy_3d_parms);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

//int CudaTexture3D::download(unsigned int size, void* h_data) {
//    const unsigned int cur_size = (unsigned int)_width*(unsigned int)_height*(unsigned int)_depth*
//        (unsigned int)CudaUtils::get_componet_byte(_channel);
//    if (size != cur_size) {
//        MI_CUDARESOURCE_LOG(MI_ERROR) << "invalid size: " << size << " when download cuda 2D surface with size: " << cur_size;
//        return -1;
//    }
//
//    cudaExtent extent = { _width, _height, _depth };
//    cudaMemcpy3DParms memcpy_3d_parms;
//    memset(&memcpy_3d_parms, 0, sizeof(cudaMemcpy3DParms));
//    memcpy_3d_parms.srcArray = _d_array;
//    memcpy_3d_parms.extent = extent;
//    memcpy_3d_parms.kind = cudaMemcpyDeviceToHost;
//    memcpy_3d_parms.dstPos.x = 0;
//    memcpy_3d_parms.dstPos.y = 0;
//    memcpy_3d_parms.dstPos.z = 0;
//    memcpy_3d_parms.srcPtr = make_cudaPitchedPtr(h_data, _width*CudaUtils::get_componet_byte(_channel), _width, _height);
//
//    cudaError_t err = cudaMemcpy3D(&memcpy_3d_parms);
//    if (err != cudaSuccess) {
//        LOG_CUDA_ERROR(err);
//        return -1;
//    }
//    else {
//        return 0;
//    }
//}

cudaTextureObject_t CudaTexture3D::create_object(
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