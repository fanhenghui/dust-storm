#include <cuda_runtime.h>

#include "mi_cuda_global_memory.h"
#include "log/mi_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaGlobalMemory::CudaGlobalMemory():_d_array(nullptr), _size(0) {

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

void CudaGlobalMemory::load(size_t size, const void* h_array) {
    if (_d_array) {
        if (_size == size) {
            if (h_array) {
                cudaMemcpy(_d_array, h_array, size , cudaMemcpyHostToDevice);
            }
            return;
        } else {
            cudaFree(_d_array);
            _d_array = nullptr;
            _size = 0;
        }
    }

    if (size == 0) {
        MI_LOG(MI_ERROR) << "can't load 0 CUDA global memory.";
        return;
    }
    cudaMalloc(&_d_array, size);
    _size = size;
    if (h_array) {
        cudaMemcpy(_d_array, h_array, size, cudaMemcpyHostToDevice);
    }
}

void CudaGlobalMemory::download(void* h_array, size_t size) {
    if (!_d_array) {
        MI_LOG(MI_ERROR) << "can't download empty CUDA global memory.";
        return;
    }
    if (size != _size) {
        MI_LOG(MI_ERROR) << "can't download CUDA global memory size " << _size << " to host array size " << size;
        return;
    }

    cudaMemcpy(h_array, _d_array, size, cudaMemcpyDeviceToHost);
}

void* CudaGlobalMemory::get_array() {
    return _d_array;
}

MED_IMG_END_NAMESPACE