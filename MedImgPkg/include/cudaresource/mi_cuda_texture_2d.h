#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_2D_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_2D_H

#include "cudaresource/mi_cuda_texture_base.h"


MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTexture2D : public CudaTextureBase {
public:
    explicit CudaTexture2D(UIDType uid);
    virtual ~CudaTexture2D();

    virtual void finalize();
    virtual float memory_used() const;

    int get_width() const;

    int get_height() const;

    int load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int width, int height, void* h_data);

    int update(int offset_x, int offset_y, int width, int height, void* h_data);

    int download(unsigned int size, void* h_data);

protected:
    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

private:
    int _width;
    int _height;
};

MED_IMG_END_NAMESPACE
#endif
