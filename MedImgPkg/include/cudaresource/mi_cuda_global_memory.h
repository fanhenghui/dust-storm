#ifndef MED_IMG_CUDARESOUECE_MI_GLOBAL_MEMORY_H
#define MED_IMG_CUDARESOUECE_MI_GLOBAL_MEMORY_H

#include "cudaresource/mi_cuda_object.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaGlobalMemory : public CudaObject {
public:
    explicit CudaGlobalMemory(UIDType uid);
    virtual ~CudaGlobalMemory();

    virtual void initialize();
    virtual void finalize();
    virtual float memory_used() const;

    size_t get_size() const;

    int load(size_t size, const void* h_array);
    int update(size_t offset, size_t size, const void* h_array);
    int download(size_t size, void* h_array);
    void* get_pointer();

private:
    void* _d_array;
    size_t _size;
private:
    DISALLOW_COPY_AND_ASSIGN(CudaGlobalMemory);
};

MED_IMG_END_NAMESPACE
#endif
