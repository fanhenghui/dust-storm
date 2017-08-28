#ifndef MED_IMG_RAY_CATING_GPU_STEP_UTILS_H
#define MED_IMG_RAY_CATING_GPU_STEP_UTILS_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE 

class RCStepUtils : public RCStepBase
{
public:
    RCStepUtils(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):RCStepBase(ray_caster , program)
    {};

    virtual ~RCStepUtils(){};

    virtual GLShaderInfo get_shader_info();

private:
};

MED_IMG_END_NAMESPACE

#endif