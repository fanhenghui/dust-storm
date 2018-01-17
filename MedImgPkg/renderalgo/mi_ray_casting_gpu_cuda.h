#ifndef MED_IMG_RENDERALGORITHM_MI_RAY_CASTING_GPU_CUDA_H
#define MED_IMG_RENDERALGORITHM_MI_RAY_CASTING_GPU_CUDA_H

#include "cudaresource/mi_cuda_resource_define.h"
#include "mi_ray_caster_cuda_define.h"

MED_IMG_BEGIN_NAMESPACE
class RayCaster;
class RayCastingGPUCUDA
{
public:
    RayCastingGPUCUDA(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingGPUCUDA();

    float get_rendering_duration() const;

    void render();

    void on_entry_exit_points_resize(int width, int height);

private:
    void fill_paramters(std::shared_ptr<RayCaster> , CudaVolumeInfos&  , CudaRayCastInfos&);

private:
    std::weak_ptr<RayCaster> _ray_caster;
    
    float _duration;

    struct InnerResource;
    std::unique_ptr<InnerResource> _inner_resource;

private:
    DISALLOW_COPY_AND_ASSIGN(RayCastingGPUCUDA);
};

MED_IMG_END_NAMESPACE

#endif
