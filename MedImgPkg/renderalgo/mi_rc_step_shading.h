#ifndef MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_SHADING_H
#define MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_SHADING_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE

class RCStepShadingNone : public RCStepBase {
public:
    RCStepShadingNone(std::shared_ptr<RayCaster> ray_caster,
                      std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepShadingNone() {};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepShadingPhong : public RCStepBase {
public:
    RCStepShadingPhong(std::shared_ptr<RayCaster> ray_caster,
                       std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program), _loc_mat_normal(-1), _loc_spacing(-1),
          _loc_light_position(-1), _loc_ambient_color(-1) {};

    virtual ~RCStepShadingPhong() {};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

    virtual void get_uniform_location();

private:
    int _loc_mat_normal;
    int _loc_spacing;
    int _loc_light_position;
    int _loc_ambient_color;
};

MED_IMG_END_NAMESPACE

#endif