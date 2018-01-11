#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_1D_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TEXTURE_1D_H

#include "cudaresource/mi_cuda_texture_base.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTexture1D : public CudaTextureBase {
public:
    explicit CudaTexture1D(UIDType uid);
    virtual ~CudaTexture1D();
    
    virtual void finalize();
    virtual float memory_used() const;

    int get_length() const;

    int load(int channel_x, int channel_y, int channel_z, int channel_w, cudaChannelFormatKind format, int length, void* h_data);

    int update(int offset, int length, void* h_data);

    int download(unsigned int size, void* h_data);

protected:

    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

private:
    int _length;
};

MED_IMG_END_NAMESPACE
#endif
