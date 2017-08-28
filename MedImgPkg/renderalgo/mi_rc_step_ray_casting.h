#ifndef MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_RAY_CASTING_H
#define MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_RAY_CASTING_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE

class RCStepRayCastingMIPBase : public RCStepBase {
public:
  RCStepRayCastingMIPBase(std::shared_ptr<RayCaster> ray_caster,
                          std::shared_ptr<GLProgram> program)
      : RCStepBase(ray_caster, program), _loc_pseudo_color(-1),
        _loc_pseudo_color_slope(-1), _loc_pseudo_color_intercept(-1),
        _loc_global_wl(-1){};

  virtual ~RCStepRayCastingMIPBase(){};

  virtual void set_gpu_parameter();

  virtual void get_uniform_location();

private:
  int _loc_pseudo_color;
  int _loc_pseudo_color_slope;
  int _loc_pseudo_color_intercept;
  int _loc_global_wl;
};

class RCStepRayCastingAverage : public RCStepRayCastingMIPBase {
public:
  RCStepRayCastingAverage(std::shared_ptr<RayCaster> ray_caster,
                          std::shared_ptr<GLProgram> program)
      : RCStepRayCastingMIPBase(ray_caster, program){};

  virtual ~RCStepRayCastingAverage(){};

  virtual GLShaderInfo get_shader_info();

private:
};

class RCStepRayCastingMIPMinIP : public RCStepRayCastingMIPBase {
public:
  RCStepRayCastingMIPMinIP(std::shared_ptr<RayCaster> ray_caster,
                           std::shared_ptr<GLProgram> program)
      : RCStepRayCastingMIPBase(ray_caster, program){};

  virtual ~RCStepRayCastingMIPMinIP(){};

  virtual GLShaderInfo get_shader_info();

private:
  ;
};

class RCStepRayCastingDVR : public RCStepBase {
public:
  RCStepRayCastingDVR(std::shared_ptr<RayCaster> ray_caster,
                      std::shared_ptr<GLProgram> program)
      : RCStepBase(ray_caster, program){};

  virtual ~RCStepRayCastingDVR(){};

  virtual GLShaderInfo get_shader_info();

private:
  ;
};

MED_IMG_END_NAMESPACE

#endif