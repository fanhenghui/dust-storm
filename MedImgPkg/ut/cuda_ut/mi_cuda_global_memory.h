#ifndef MI_CUDA_GLOBAL_MEMMORY_H
#define MI_CUDA_GLOBAL_MEMMORY_H

#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE

class CudaGlobalMemory
{
public:
    CudaGlobalMemory();
    ~CudaGlobalMemory();

    void initialize();
    void finalize();
    void load(size_t size, const void* h_array);
    void download(void* h_array, size_t size);
    void* get_array();
    
protected:
private:
    void* _d_array;
    size_t _size;
};

MED_IMG_END_NAMESPACE
#endif