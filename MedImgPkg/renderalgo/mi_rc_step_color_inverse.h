#ifndef MED_IMG_RAY_CATING_GPU_STEP_COLOR_INVERSE_H_
#define MED_IMG_RAY_CATING_GPU_STEP_COLOR_INVERSE_H_

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE 

class RCStepColorInverseEnable : public RCStepBase
{
public:
    RCStepColorInverseEnable(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):
      RCStepBase(ray_caster , program)
      {};

      virtual ~RCStepColorInverseEnable(){};

      virtual GLShaderInfo get_shader_info();

private:
};

class RCStepColorInverseDisable : public RCStepBase
{
public:
    RCStepColorInverseDisable(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):
      RCStepBase(ray_caster , program)
      {};

      virtual ~RCStepColorInverseDisable(){};

      virtual GLShaderInfo get_shader_info();

private:
};

MED_IMG_END_NAMESPACE
#endif