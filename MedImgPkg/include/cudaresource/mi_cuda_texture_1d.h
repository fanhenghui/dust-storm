#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_1D_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_1D_H

#include "cudaresource/mi_cuda_texture_base.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTexture1D : public CUDATextureBase
{
public:
    explicit CudaTexture1D(UIDType uid);
    virtual ~CudaTexture1D();
    
    virtual void finalize();
    virtual float memory_used() const;

    int load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int length, void* data);

    int update(int offset, int length, void* data);

protected:

    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

private:
    int _length;
    int _channel[4];
    cudaChannelFormatKind _format;
};

MED_IMG_END_NAMESPACE
#endif
