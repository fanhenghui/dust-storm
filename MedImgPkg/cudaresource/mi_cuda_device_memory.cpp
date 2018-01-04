#include "mi_cuda_device_memory.h"
#include <cuda_runtime.h>
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaDeviceMemory::CudaDeviceMemory(UIDType uid) :CudaObject(uid, "CUDADeviceMemory"), _d_array(nullptr), _size(0) {
}

CudaDeviceMemory::~CudaDeviceMemory() {
    finalize();
}

void CudaDeviceMemory::initialize() {

}

void CudaDeviceMemory::finalize() {
    if (_d_array) {
        cudaFree(_d_array);
        _d_array = nullptr;
        _size = 0;
    }
}

float CudaDeviceMemory::memory_used() const {
    if (_d_array) {
        return _size/1024.0f;
    } else {
        return 0.0f;
    }
}

void CudaDeviceMemory::load(size_t size, const void* h_array) {
    if (_d_array) {
        if (_size == size) {
            if (h_array) {
                cudaMemcpy(_d_array, h_array, size, cudaMemcpyHostToDevice);
            }
            return;
        } else {
            cudaFree(_d_array);
            _d_array = nullptr;
            _size = 0;
        }
    }

    if (size == 0) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "can't load 0 CUDA global memory.";
        return;
    }
    cudaMalloc(&_d_array, size);
    _size = size;
    if (h_array) {
        cudaMemcpy(_d_array, h_array, size, cudaMemcpyHostToDevice);
    }
}

void CudaDeviceMemory::download(void* h_array, size_t size) {
    if (!_d_array) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "can't download empty CUDA global memory.";
        return;
    }
    if (size != _size) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "can't download CUDA global memory size " << _size << " to host array size " << size;
        return;
    }

    cudaMemcpy(h_array, _d_array, size, cudaMemcpyDeviceToHost);
}

void* CudaDeviceMemory::get_pointer() {
    return _d_array;
}

MED_IMG_END_NAMESPACE