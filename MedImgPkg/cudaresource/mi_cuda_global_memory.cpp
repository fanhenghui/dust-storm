#include "mi_cuda_global_memory.h"
#include <cuda_runtime.h>
#include "mi_cuda_resource_logger.h"
#include "mi_cuda_utils.h"

MED_IMG_BEGIN_NAMESPACE

CudaGlobalMemory::CudaGlobalMemory(UIDType uid) :CudaObject(uid, "CUDADeviceMemory"), _d_array(nullptr), _size(0) {
}

CudaGlobalMemory::~CudaGlobalMemory() {
    finalize();
}

void CudaGlobalMemory::initialize() {

}

void CudaGlobalMemory::finalize() {
    if (_d_array) {
        cudaFree(_d_array);
        _d_array = nullptr;
        _size = 0;
    }
}

float CudaGlobalMemory::memory_used() const {
    if (_d_array) {
        return _size/1024.0f;
    } else {
        return 0.0f;
    }
}

int CudaGlobalMemory::load(size_t size, const void* h_array) {
    cudaError_t err = cudaSuccess;
    if (_d_array) {
        if (_size == size) {
            if (h_array) {
                err = cudaMemcpy(_d_array, h_array, size, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_CUDA_ERROR(err);
                    return -1;
                }
                return 0;
            }
        } else {
            err = cudaFree(_d_array);
            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                return -1;
            }
            _d_array = nullptr;
            _size = 0;
        }
    }

    if (size == 0) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "can't load 0 CUDA global memory.";
        return -1;
    }
    err = cudaMalloc(&_d_array, size);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }

    _size = size;
    if (h_array) {
        err = cudaMemcpy(_d_array, h_array, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            return -1;
        }
    }

    return 0;
}

int CudaGlobalMemory::download(size_t size, void* h_array) {
    if (!_d_array) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "can't download empty CUDA global memory.";
        return -1;
    }
    if (size != _size) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "can't download CUDA global memory size " << _size << " to host array size " << size;
        return -1;
    }

    cudaError_t err = cudaMemcpy(h_array, _d_array, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }

    return 0;
}

void* CudaGlobalMemory::get_pointer() {
    return _d_array;
}

MED_IMG_END_NAMESPACE