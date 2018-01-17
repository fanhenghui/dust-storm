#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_GL_INTEROP_CACHE_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_GL_INTEROP_CACHE_H

#include "cudaresource/mi_cuda_resource_export.h"
#include <boost/thread/mutex.hpp>

#include <list>

#include "cudaresource/mi_cuda_resource_define.h"

MED_IMG_BEGIN_NAMESPACE

class CudaGLInteropCache {
public:
    ~CudaGLInteropCache();
    
    static CudaGLInteropCache* instance();

    void cache_interop_unregister_gl_texture(CudaGLTexture2DPtr tex);

    void process_cache();
private:
    CudaGLInteropCache();
    static CudaGLInteropCache* _s_instance;
    static boost::mutex _s_mutex;

private:
    boost::mutex _mutex;
    std::list<std::weak_ptr<CudaGLTexture2D>> _interop_list_unreg_gl_tex;
private:
    DISALLOW_COPY_AND_ASSIGN(CudaGLInteropCache);
};

MED_IMG_END_NAMESPACE

#endif