#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_GL_TEXTURE_1D_ARRAY_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_GL_TEXTURE_1D_ARRAY_H

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include "cudaresource/mi_cuda_object.h"
#include "cudaresource/mi_cuda_texture_1d.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTexture1DArray : public CudaObject {
public:
    explicit CudaTexture1DArray(UIDType uid, const int array_length);

    virtual ~CudaTexture1DArray();

    virtual void initialize();

    virtual void finalize();

    virtual float memory_used() const;
    
    int get_length() const;

    int load(int channel_x, int channel_y, int channel_z, int channel_w, int position, cudaChannelFormatKind format, int length, void* data);

    void get_texture_array(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords, cudaTextureObject_t* tex_array);

private:
    cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords, cudaArray_t cuda_array);

private:
    int _length;
    int _channel[4];
    cudaChannelFormatKind _format;

    int _array_length;
    std::map<int, cudaArray_t> _cuda_arrays;
    std::map<int, std::map<int, cudaTextureObject_t>> _tex_objs;
};

MED_IMG_END_NAMESPACE



#endif
