#ifndef MED_IMG_CUDARESOUECE_MI_DEVICE_MEMORY_H
#define MED_IMG_CUDARESOUECE_MI_DEVICE_MEMORY_H

#include "cudaresource/mi_cuda_object.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaDeviceMemory : public CudaObject
{
public:
    explicit CudaDeviceMemory(UIDType uid);
    virtual ~CudaDeviceMemory();

    virtual void initialize();
    virtual void finalize();
    virtual float memory_used() const;

    void load(size_t size, const void* h_array);
    void download(void* h_array, size_t size);
    void* get_pointer();

private:
    void* _d_array;
    size_t _size;
private:
    DISALLOW_COPY_AND_ASSIGN(CudaDeviceMemory);
};

MED_IMG_END_NAMESPACE
#endif
