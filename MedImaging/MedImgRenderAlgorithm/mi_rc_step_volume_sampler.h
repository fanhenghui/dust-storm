#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_VOLUME_SAMPLER_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_VOLUME_SAMPLER_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepVolumeNearstSampler : public RCStepBase
{
public:
    RCStepVolumeNearstSampler(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepVolumeNearstSampler(){};

    virtual GLShaderInfo GetShaderInfo();

private:
};

class RCStepVolumeLinearSampler : public RCStepBase
{
public:
    RCStepVolumeLinearSampler(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepVolumeLinearSampler(){};

    virtual GLShaderInfo GetShaderInfo();

private:
};

MED_IMAGING_END_NAMESPACE

#endif