#ifndef MED_IMG_RENDERALGO_MI_RENDER_GLOBAL_RESOURCE_H
#define MED_IMG_RENDERALGO_MI_RENDER_GLOBAL_RESOURCE_H

#include "renderalgo/mi_gpu_resource_pair.h"
#include "renderalgo/mi_ray_caster_define.h"

#include <boost/thread/mutex.hpp>

MED_IMG_BEGIN_NAMESPACE

class RenderGlobalResource {
public:
    ~RenderGlobalResource();
    static RenderGlobalResource* instance(GPUPlatform p);

    GPUTexture2DPairPtr get_random_texture();
private:
    RenderGlobalResource(GPUPlatform platform);
    static boost::mutex _s_mutex;
    static RenderGlobalResource* _s_instance_gl;
    static RenderGlobalResource* _s_instance_cuda;

    boost::mutex _mutex;
    GPUPlatform _gpu_platform;
    GPUTexture2DPairPtr _random_texture;
    
private:
    DISALLOW_COPY_AND_ASSIGN(RenderGlobalResource);
};

MED_IMG_END_NAMESPACE

#endif