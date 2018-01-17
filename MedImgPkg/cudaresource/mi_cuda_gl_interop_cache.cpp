#include "mi_cuda_gl_interop_cache.h"
#include "mi_cuda_gl_texture_2d.h"

#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaGLInteropCache* CudaGLInteropCache::_s_instance = nullptr;
boost::mutex CudaGLInteropCache::_s_mutex;

CudaGLInteropCache::CudaGLInteropCache() {

}

CudaGLInteropCache::~CudaGLInteropCache() {

}

CudaGLInteropCache* CudaGLInteropCache::instance() {
    if (nullptr == _s_instance) {
        boost::mutex::scoped_lock locker(_s_mutex);
        if (nullptr == _s_instance) {
            _s_instance = new CudaGLInteropCache(); 
        }
    }
    return _s_instance;
}

void CudaGLInteropCache::cache_interop_unregister_gl_texture(CudaGLTexture2DPtr tex) {
    boost::mutex::scoped_lock locker(_mutex);
    _interop_list_unreg_gl_tex.push_back(tex);
}

void CudaGLInteropCache::process_cache() {
    boost::mutex::scoped_lock locker(_mutex);
    for (auto it = _interop_list_unreg_gl_tex.begin(); it != _interop_list_unreg_gl_tex.end(); ++it) {
        CudaGLTexture2DPtr cuda_gl_tex = (*it).lock();
        if (cuda_gl_tex) {
            cuda_gl_tex->unregister_gl_texture();
        }
    }
    _interop_list_unreg_gl_tex.clear();
}

MED_IMG_END_NAMESPACE