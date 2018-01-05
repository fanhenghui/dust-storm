#include "mi_cuda_texture_base.h"
#include "mi_cuda_resource_logger.h"
#include "mi_cuda_utils.h"

MED_IMG_BEGIN_NAMESPACE

CudaTextureBase::CudaTextureBase(UIDType uid, const std::string& type): 
    CudaObject(uid, type), _d_array(nullptr), _format(cudaChannelFormatKindNone){
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
}

CudaTextureBase::~CudaTextureBase() {
    finalize();
}

void CudaTextureBase::initialize() {

}

void CudaTextureBase::finalize() {
    cudaError_t err = cudaSuccess;
    if (_d_array) {
        err = cudaFreeArray(_d_array);
        CHECK_CUDA_ERROR(err);
        _d_array = nullptr;
    }
    for (auto it = _tex_objs.begin(); it != _tex_objs.end(); ++it) {
        err = cudaDestroyTextureObject(it->second);
        CHECK_CUDA_ERROR(err);
    }
    _tex_objs.clear();
}

void CudaTextureBase::get_channel(int(&channel)[4]) const {
    memcpy(channel, _channel, sizeof(int) * 4);
}

cudaTextureObject_t CudaTextureBase::get_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode, bool normalized_coords) {
    //cudaTextureAddressMode : bit 7~ ...
    //cudaTextureFilterMode : bit 4~6
    //cudaTextureReadMode : bit 1~3
    //normalized Coordinate : bit 0
    int id = 0;
    id |= (int(address_mode)) << 7;
    id |= (int(filter_mode)) << 4;
    id |= (int(read_mode)) << 1;
    id |= (int(normalized_coords));
    auto it = _tex_objs.find(id);
    if (it != _tex_objs.end()) {
        return it->second;
    } else {
        if (nullptr == _d_array) {
            MI_CUDARESOURCE_LOG(MI_ERROR) << "try get null CUDA array's texture object.";
            return 0;
        }

        //create new texture obj
        cudaTextureObject_t tex_obj = this->create_object(address_mode, filter_mode, read_mode, normalized_coords);
        _tex_objs.insert(std::make_pair(id, tex_obj));

        return tex_obj;
    }
}

MED_IMG_END_NAMESPACE