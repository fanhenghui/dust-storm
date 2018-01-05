#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_UTILS_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_UTILS_H

#include "cudaresource/mi_cuda_resource_export.h"
#include <cuda_runtime.h>

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaUtils {
public:
    static void log_cuda_error(cudaError_t err,  const char* func, const char* file, const int line);
    static int get_componet_byte(const int (&channel)[4]);
};

#define LOG_CUDA_ERROR(err) {CudaUtils::log_cuda_error(err, __FUNCTION__, __FILE__, __LINE__); }

#define CHECK_CUDA_ERROR(err) {\
    if (err != cudaSuccess) {\
        CudaUtils::log_cuda_error(err, __FUNCTION__, __FILE__, __LINE__); \
}}\

#define CHECK_LAST_CUDA_ERROR {\
cudaError_t err = cudaGetLastError(); \
CHECK_CUDA_ERROR(err) \
}\ 

MED_IMG_END_NAMESPACE

#endif 
