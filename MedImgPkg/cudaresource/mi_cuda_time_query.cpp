#include "mi_cuda_time_query.h"
#include "mi_cuda_utils.h"

MED_IMG_BEGIN_NAMESPACE

CudaTimeQuery::CudaTimeQuery(UIDType uid) : 
CudaObject(uid, "CudaTimeQuery"), _start(nullptr), _end(nullptr), _time_elapsed(0.0f) {

}

CudaTimeQuery::~CudaTimeQuery() {

}

void CudaTimeQuery::initialize() {
    if (nullptr == _start) {
        cudaError err = cudaEventCreate(&_start);
        CHECK_CUDA_ERROR(err);
    }

    if (nullptr == _end) {
        cudaError err = cudaEventCreate(&_end);
        CHECK_CUDA_ERROR(err);
    }
}

void CudaTimeQuery::finalize() {
    if (nullptr != _start) {
        cudaError err = cudaEventDestroy(_start);
        CHECK_CUDA_ERROR(err);
        _start = nullptr;
    }

    if (nullptr != _end) {
        cudaError err = cudaEventDestroy(_end);
        CHECK_CUDA_ERROR(err);
        _end = nullptr;
    }

    _time_elapsed = 0.0f;
}

float CudaTimeQuery::memory_used() const {
    return 0.0f;
}

void CudaTimeQuery::begin() {
    initialize();
    cudaEventRecord(_start, 0);
}

float CudaTimeQuery::end() {
    initialize();
    cudaEventRecord(_end, 0);
    cudaEventSynchronize(_end);
    cudaEventElapsedTime(&_time_elapsed, _start, _end);
    return _time_elapsed;
}

float CudaTimeQuery::get_time_elapsed() const {
    return _time_elapsed;
}

MED_IMG_END_NAMESPACE