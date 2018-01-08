#ifndef MED_IMG_RENDERALGORITHM_MI_RAY_CASTING_GPU_CUDA_H
#define MED_IMG_RENDERALGORITHM_MI_RAY_CASTING_GPU_CUDA_H

#include "cudaresource/mi_cuda_resource_define.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE
class RayCaster;
class RayCastingGPUCUDA
{
public:
    RayCastingGPUCUDA(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingGPUCUDA();

    double get_rendering_duration() const;

    void render();

private:
    std::weak_ptr<RayCaster> _ray_caster;
};

MED_IMG_END_NAMESPACE

#endif
