#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_BASE_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_BASE_H

#include <map>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include "cudaresource/mi_cuda_object.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CUDATextureBase : public CudaObject
{
public:
    CUDATextureBase(UIDType uid, const std::string& type);
    virtual ~CUDATextureBase();

    virtual void initialize();
    virtual void finalize();

    cudaTextureObject_t get_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

protected:
    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode, 
        cudaTextureReadMode read_mode, bool normalized_coords) = 0;

protected:
    cudaArray_t _cuda_array;
    std::map<int, cudaTextureObject_t> _tex_objs;

private:
    DISALLOW_COPY_AND_ASSIGN(CUDATextureBase);
};

MED_IMG_END_NAMESPACE

#endif