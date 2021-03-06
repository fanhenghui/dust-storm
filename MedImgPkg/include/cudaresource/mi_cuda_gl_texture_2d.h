#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_GL_TEXTURE_2D_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_GL_TEXTURE_2D_H

#include "cudaresource/mi_cuda_texture_base.h"
#include "GL/glew.h"
#include <cuda_gl_interop.h>

MED_IMG_BEGIN_NAMESPACE

class GLTexture2D;
class CUDAResource_Export CudaGLTexture2D : public CudaTextureBase {
public:
    explicit CudaGLTexture2D(UIDType uid);
    virtual ~CudaGLTexture2D();

    virtual void finalize();
    virtual float memory_used() const;

    int register_gl_texture(std::shared_ptr<GLTexture2D> tex, cudaGraphicsRegisterFlags register_flag);

    int unregister_gl_texture();

    bool valid();

    int map_gl_texture();

    int unmap_gl_texture();

    int write_to_gl_texture(void* array, size_t count, cudaMemcpyKind memcpy_kind);

    int get_width() const;

    int get_height() const;

    int download(unsigned int size, void* h_data);

protected:
    virtual cudaTextureObject_t create_object(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
        cudaTextureReadMode read_mode, bool normalized_coords);

private:
    int _width;
    int _height;

    //GL Interoperability
    cudaGraphicsResource_t _cuda_graphic_resource;
    cudaGraphicsRegisterFlags _register_flag;
};

MED_IMG_END_NAMESPACE
#endif
