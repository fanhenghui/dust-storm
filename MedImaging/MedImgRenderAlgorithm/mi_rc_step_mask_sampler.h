#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_MASK_SAMPLER_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_MASK_SAMPLER_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepMaskNoneSampler : public RCStepBase
{
public:
    RCStepMaskNoneSampler(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepMaskNoneSampler(){};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepMaskNearstSampler : public RCStepBase
{
public:
    RCStepMaskNearstSampler(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepMaskNearstSampler(){};

    virtual GLShaderInfo get_shader_info();

private:
};

MED_IMAGING_END_NAMESPACE

#endif