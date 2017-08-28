#ifndef MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_MAIN_H
#define MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_MAIN_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE

class RCStepMainVert : public RCStepBase {
public:
  RCStepMainVert(std::shared_ptr<RayCaster> ray_caster,
                 std::shared_ptr<GLProgram> program)
      : RCStepBase(ray_caster, program){};

  virtual ~RCStepMainVert(){};

  virtual GLShaderInfo get_shader_info();

  virtual void set_gpu_parameter();

private:
};

class RCStepMainFrag : public RCStepBase {
public:
  RCStepMainFrag(std::shared_ptr<RayCaster> ray_caster,
                 std::shared_ptr<GLProgram> program)
      : RCStepBase(ray_caster, program), _loc_volume_dim(-1),
        _loc_volume_data(-1), _loc_mask_data(-1), _loc_sample_rate(-1){

                                                  };

  virtual ~RCStepMainFrag(){};

  virtual GLShaderInfo get_shader_info();

  virtual void set_gpu_parameter();

  virtual void get_uniform_location();

private:
  int _loc_volume_dim;
  int _loc_volume_data;
  int _loc_mask_data;
  int _loc_sample_rate;
};

class RCStepMainTestFrag : public RCStepBase {
public:
  RCStepMainTestFrag(std::shared_ptr<RayCaster> ray_caster,
                     std::shared_ptr<GLProgram> program)
      : RCStepBase(ray_caster, program), _loc_volume_dim(-1),
        _loc_test_code(-1){

        };

  virtual ~RCStepMainTestFrag(){};

  virtual GLShaderInfo get_shader_info();

  virtual void set_gpu_parameter();

  virtual void get_uniform_location();

private:
  int _loc_volume_dim;
  int _loc_test_code;
};

MED_IMG_END_NAMESPACE

#endif