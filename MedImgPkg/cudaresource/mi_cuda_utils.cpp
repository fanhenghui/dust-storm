#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

void CudaUtils::log_cuda_error(cudaError_t err, const char* func, const char* file, const int line) {
    MI_CUDARESOURCE_LOG(MI_ERROR) << "catch CUDA error ID: " << err << "in file: " << file << ", line: " << line << ", func: " << func;
}

int CudaUtils::get_component_byte(const int(&channel)[4]) {
    int x = 0;
    x += channel[0]>0 ? channel[0] / 8 : 0;
    x += channel[1]>0 ? channel[1] / 8 : 0;
    x += channel[2]>0 ? channel[2] / 8 : 0;
    x += channel[3]>0 ? channel[3] / 8 : 0;
   return x;
}

MED_IMG_END_NAMESPACE

