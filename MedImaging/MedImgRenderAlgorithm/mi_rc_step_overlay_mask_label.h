#ifndef MED_IMAGING_RAY_CATING_GPU_STEP_OVERLAY_MASK_LABEL_H_
#define MED_IMAGING_RAY_CATING_GPU_STEP_OVERLAY_MASK_LABEL_H_

#include "mi_rc_step_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class RCStepOverlayMaskLabelEnable : public RCStepBase
{
public:
    RCStepOverlayMaskLabelEnable(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):
      RCStepBase(ray_caster , program),_loc_visible_label_count(-1)
      {};

      virtual ~RCStepOverlayMaskLabelEnable(){};

      virtual GLShaderInfo get_shader_info();

      virtual void get_uniform_location();

      virtual void set_gpu_parameter();

private:
    int _loc_visible_label_count;
};

class RCStepOverlayMaskLabelDisable : public RCStepBase
{
public:
    RCStepOverlayMaskLabelDisable(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program):
      RCStepBase(ray_caster , program)
      {};

      virtual ~RCStepOverlayMaskLabelDisable(){};

      virtual GLShaderInfo get_shader_info();

private:
};

MED_IMAGING_END_NAMESPACE
#endif