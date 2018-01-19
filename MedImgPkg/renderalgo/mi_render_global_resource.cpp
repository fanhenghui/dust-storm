#include "mi_render_global_resource.h"

#include <random>

#include "util/mi_memory_shield.h"

#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"

#include "cudaresource/mi_cuda_resource_manager.h"
#include "cudaresource/mi_cuda_texture_2d.h"


MED_IMG_BEGIN_NAMESPACE

RenderGlobalResource* RenderGlobalResource::_s_instance_gl = nullptr;
RenderGlobalResource* RenderGlobalResource::_s_instance_cuda = nullptr;
boost::mutex RenderGlobalResource::_s_mutex;

RenderGlobalResource::RenderGlobalResource(GPUPlatform platform):_gpu_platform(platform) {

}

RenderGlobalResource::~RenderGlobalResource() {

}

RenderGlobalResource* RenderGlobalResource::instance(GPUPlatform platform) {
    if (GL_BASE == platform) {
        if (nullptr == _s_instance_gl) {
            boost::mutex::scoped_lock locker(_s_mutex);
            if (nullptr == _s_instance_gl) {
                _s_instance_gl = new RenderGlobalResource(platform);
            }
        }
        return _s_instance_gl;

    } else {
        if (nullptr == _s_instance_cuda) {
            boost::mutex::scoped_lock locker(_s_mutex);
            if (nullptr == _s_instance_cuda) {
                _s_instance_cuda = new RenderGlobalResource(platform);
            }
        }
        return _s_instance_cuda;
    }
}

GPUTexture2DPairPtr RenderGlobalResource::get_random_texture() {
    if (nullptr != _random_texture) {
        return _random_texture;
    }

    boost::mutex::scoped_lock locker(_mutex);

    //random map 
    const int size = 32;
    srand((unsigned int)time(0));
    unsigned short* data = new unsigned short[size*size];
    for (int i = 0; i < size*size; ++i) {
        data[i] = (unsigned short)((rand() / (float)RAND_MAX) * 65535.0f * 0.5f);
    }

    //random texture 
    if (GL_BASE == _gpu_platform) {
        GLTexture2DPtr tex = GLResourceManagerContainer::instance()->
            get_texture_2d_manager()->create_object("global random texture");
        _random_texture.reset(new GPUTexture2DPair(tex));
        GLTextureCache::instance()->cache_load(GL_TEXTURE_2D, tex, 
            GL_CLAMP_TO_BORDER, GL_LINEAR, GL_R16, size, size, 1, GL_RED, GL_UNSIGNED_SHORT, data);
    } else {
        MemShield mem_shield(data);
        CudaTexture2DPtr tex = CudaResourceManager::instance()->
            create_cuda_texture_2d("global random texture");
        _random_texture.reset(new GPUTexture2DPair(tex));
        tex->load(16,0,0,0, cudaChannelFormatKindUnsigned, size, size, data);
    }

    return _random_texture;
}


MED_IMG_END_NAMESPACE