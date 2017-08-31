#ifndef MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_MASK_SAMPLER_H
#define MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_MASK_SAMPLER_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE

class RCStepMaskNoneSampler : public RCStepBase {
public:
    RCStepMaskNoneSampler(std::shared_ptr<RayCaster> ray_caster,
                          std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepMaskNoneSampler() {};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepMaskNearstSampler : public RCStepBase {
public:
    RCStepMaskNearstSampler(std::shared_ptr<RayCaster> ray_caster,
                            std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepMaskNearstSampler() {};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

private:
};

class RCStepMaskLinearSampler : public RCStepBase {
public:
    RCStepMaskLinearSampler(std::shared_ptr<RayCaster> ray_caster,
        std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program), _loc_visible_label_count(-1) {};

    virtual ~RCStepMaskLinearSampler() {};

    virtual GLShaderInfo get_shader_info();

    virtual void get_uniform_location();

    virtual void set_gpu_parameter();

private:
    int _loc_visible_label_count;
};

MED_IMG_END_NAMESPACE

#endif