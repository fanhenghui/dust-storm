#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_UTILS_H
#define MED_IMAGING_RAY_CATING_GPU_STEP_UTILS_H

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepUtils : public RCStepBase
{
public:
    RCStepUtils(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram):RCStepBase(pRayCaster , pProgram)
    {};

    virtual ~RCStepUtils(){};

    virtual GLShaderInfo GetShaderInfo();

private:
};

MED_IMAGING_END_NAMESPACE

#endif