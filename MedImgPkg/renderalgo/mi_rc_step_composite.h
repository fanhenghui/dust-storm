#ifndef MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_COMPOSITE_H
#define MEDIMGRENDERALGO_RAY_CATING_GPU_STEP_COMPOSITE_H

#include "mi_rc_step_base.h"

MED_IMG_BEGIN_NAMESPACE

class RCStepCompositeAverage : public RCStepBase {
public:
    RCStepCompositeAverage(std::shared_ptr<RayCaster> ray_caster,
                           std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepCompositeAverage() {};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepCompositeMIP : public RCStepBase {
public:
    RCStepCompositeMIP(std::shared_ptr<RayCaster> ray_caster,
                       std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program) {};

    virtual ~RCStepCompositeMIP() {};

    virtual GLShaderInfo get_shader_info();

private:
};

class RCStepCompositeMinIP : public RCStepBase {
public:
    RCStepCompositeMinIP(std::shared_ptr<RayCaster> ray_caster,
                         std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program), _loc_custom_min_threshold(-1) {};

    virtual ~RCStepCompositeMinIP() {};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

    virtual void get_uniform_location();

private:
    int _loc_custom_min_threshold;
};

class RCStepCompositeDVR : public RCStepBase {
public:
    RCStepCompositeDVR(std::shared_ptr<RayCaster> ray_caster,
                       std::shared_ptr<GLProgram> program)
        : RCStepBase(ray_caster, program), _loc_color_opacity_array(-1),
          _loc_color_opacity_texture_shift(-1), _loc_opacity_correction(-1),
          _loc_sample_step(-1) {};

    virtual ~RCStepCompositeDVR() {};

    virtual GLShaderInfo get_shader_info();

    virtual void set_gpu_parameter();

    virtual void get_uniform_location();

private:
    int _loc_color_opacity_array;
    int _loc_color_opacity_texture_shift;
    int _loc_opacity_correction;
    int _loc_sample_step;
};

MED_IMG_END_NAMESPACE

#endif