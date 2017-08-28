#ifndef MED_IMG_RAY_CATING_GPU_STEP_MASK_SAMPLER_H
#define MED_IMG_RAY_CATING_GPU_STEP_MASK_SAMPLER_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE 

class RCStepMaskNoneSampler : public RCStepBase
{
public:
    RCStepMaskNoneSampler(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):RCStepBase(ray_caster , program)
    {};

    virtual ~RCStepMaskNoneSampler(){};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepMaskNearstSampler : public RCStepBase
{
public:
    RCStepMaskNearstSampler(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):RCStepBase(ray_caster , program)
    {};

    virtual ~RCStepMaskNearstSampler(){};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

private:
};

MED_IMG_END_NAMESPACE

#endif