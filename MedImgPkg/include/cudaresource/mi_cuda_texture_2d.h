#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_2D_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_2D_H

#include "cudaresource/mi_cuda_texture_base.h"


MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTexture2D : public CUDATextureBase
{
public:
    explicit CudaTexture2D(UIDType uid);
    virtual ~CudaTexture2D();

    virtual void finalize();

    int load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, void* data);

    int update(int offset_x, int offset_y, int width, int height, void* data);

protected:
    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

private:
    int _width;
    int _height;
    int _channel[4];
    cudaChannelFormatKind _format;
};

MED_IMG_END_NAMESPACE
#endif
