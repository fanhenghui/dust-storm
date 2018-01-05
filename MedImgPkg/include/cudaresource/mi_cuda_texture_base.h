#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_BASE_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_BASE_H

#include <map>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include "cudaresource/mi_cuda_object.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTextureBase : public CudaObject {
public:
    CudaTextureBase(UIDType uid, const std::string& type);
    virtual ~CudaTextureBase();

    virtual void initialize();
    virtual void finalize();

    cudaTextureObject_t get_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

    void get_channel(int(&channel)[4]) const;

protected:
    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode, 
        cudaTextureReadMode read_mode, bool normalized_coords) = 0;

protected:
    cudaArray_t _d_array;
    std::map<int, cudaTextureObject_t> _tex_objs;

    int _channel[4];
    cudaChannelFormatKind _format;

private:
    DISALLOW_COPY_AND_ASSIGN(CudaTextureBase);
};

MED_IMG_END_NAMESPACE

#endif