#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_SHADING_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_SHADING_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepShadingNone : public RCStepBase
{
public:
    RCStepShadingNone(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepShadingNone(){};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepShadingPhong : public RCStepBase
{
public:
    RCStepShadingPhong(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepShadingPhong(){};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

private:
};

MED_IMAGING_END_NAMESPACE

#endif