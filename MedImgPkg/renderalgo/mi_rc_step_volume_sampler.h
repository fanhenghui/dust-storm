#ifndef MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_VOLUME_SAMPLER_H
#define MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_VOLUME_SAMPLER_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE

class RCStepVolumeNearstSampler : public RCStepBase {
public:
    RCStepVolumeNearstSampler(std::shared_ptr<RayCaster> ray_caster,
                              std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepVolumeNearstSampler() {};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepVolumeLinearSampler : public RCStepBase {
public:
    RCStepVolumeLinearSampler(std::shared_ptr<RayCaster> ray_caster,
                              std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepVolumeLinearSampler() {};

    virtual GLShaderInfo get_shader_info();

private:
};

MED_IMG_END_NAMESPACE

#endif