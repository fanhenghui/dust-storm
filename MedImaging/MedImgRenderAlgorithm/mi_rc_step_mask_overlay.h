#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_MASK_OVERLAY_H_
#define MED_IMAGING_RAY_CATING_GPU_STEP_MASK_OVERLAY_H_

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepMaskOverlayEnable : public RCStepBase
{
public:
    RCStepMaskOverlayEnable(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):
      RCStepBase(ray_caster , program),_loc_visible_label_count(-1)
      {};

      virtual ~RCStepMaskOverlayEnable(){};

      virtual GLShaderInfo get_shader_info();

      virtual void get_uniform_location();

      virtual void set_gpu_parameter();

private:
    int _loc_visible_label_count;
};

class RCStepMaskOverlayDisable : public RCStepBase
{
public:
    RCStepMaskOverlayDisable(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):
      RCStepBase(ray_caster , program)
      {};

      virtual ~RCStepMaskOverlayDisable(){};

      virtual GLShaderInfo get_shader_info();

private:
};

MED_IMAGING_END_NAMESPACE
#endif