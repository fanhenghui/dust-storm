#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_SURFACE_BASE_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_SURFACE_BASE_H

#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include "cudaresource/mi_cuda_object.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaSurfaceBase : public CudaObject {
public:
    CudaSurfaceBase(UIDType uid, const std::string& type);
    virtual ~CudaSurfaceBase();

    virtual void initialize();
    virtual void finalize();

    void get_channel(int(&channel)[4]) const;

    cudaSurfaceObject_t get_object();

protected:
    cudaArray_t _d_array;
    cudaSurfaceObject_t _surface_obj;

    int _channel[4];
    cudaChannelFormatKind _format;
};

MED_IMG_END_NAMESPACE
#endif
