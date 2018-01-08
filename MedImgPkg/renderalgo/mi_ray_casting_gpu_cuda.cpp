#include "mi_ray_casting_gpu_cuda.h"
#include "mi_ray_caster.h"
#include <cuda_runtime.h>

MED_IMG_BEGIN_NAMESPACE

RayCastingGPUCUDA::RayCastingGPUCUDA(std::shared_ptr<RayCaster> ray_caster):_ray_caster(ray_caster){

}

RayCastingGPUCUDA::~RayCastingGPUCUDA() {

}

void RayCastingGPUCUDA::render() {
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster);

    const MaskMode mask_mode = ray_caster->get_mask_mode();
    const CompositeMode composite_mode = ray_caster->get_composite_mode();
    const InterpolationMode interpolation_mode = ray_caster->get_interpolation_mode();
    const ShadingMode shading_mode = ray_caster->get_shading_mode();
    const ColorInverseMode color_inverse_mode = ray_caster->get_color_inverse_mode();
    const MaskOverlayMode mask_overlay_mode = ray_caster->get_mask_overlay_mode();

    
}

double RayCastingGPUCUDA::get_rendering_duration() const {

}

MED_IMG_END_NAMESPACE