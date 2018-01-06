#include "mi_cuda_surface_base.h"
#include "mi_cuda_resource_logger.h"
#include "mi_cuda_utils.h"

MED_IMG_BEGIN_NAMESPACE

CudaSurfaceBase::CudaSurfaceBase(UIDType uid, const std::string& type) : 
    CudaObject(uid, type), _d_array(nullptr), _surface_obj(0), _format(cudaChannelFormatKindNone){
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
}

CudaSurfaceBase::~CudaSurfaceBase() {
    finalize();
}

void CudaSurfaceBase::initialize() {

}

void CudaSurfaceBase::finalize() {
    cudaError_t err = cudaSuccess;
    if (_d_array) {
        err = cudaFreeArray(_d_array);
        CHECK_CUDA_ERROR(err);
        _d_array = nullptr;
    }
    err = cudaDestroySurfaceObject(_surface_obj);
    CHECK_CUDA_ERROR(err);
}

void CudaSurfaceBase::get_channel(int(&channel)[4]) const {
    memcpy(channel, _channel, sizeof(int) * 4);
}

cudaSurfaceObject_t CudaSurfaceBase::get_object() {
    if (nullptr == _d_array) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "try get null CUDA array's surface object.";
        return 0;
    }
    if (0 == _surface_obj) {
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(cudaResourceDesc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = _d_array;
        
        cudaError_t err = cudaCreateSurfaceObject(&_surface_obj, &res_desc);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            _surface_obj = 0;
            return 0;
        } else {
            return _surface_obj;
        }
    } else {
        return _surface_obj;
    }
}

MED_IMG_END_NAMESPACE

