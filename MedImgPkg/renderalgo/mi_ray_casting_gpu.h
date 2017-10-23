#ifndef MEDIMGRENDERALGO_RAY_CASTING_GPU_H
#define MEDIMGRENDERALGO_RAY_CASTING_GPU_H

#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_render_algo_export.h"

MED_IMG_BEGIN_NAMESPACE
class RayCaster;
class RCStepBase;
class GLActiveTextureCounter;
class RayCastingGPU {
public:
    RayCastingGPU(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingGPU();

    double get_rendering_duration() const;

    void render();

private:
    void update_i();

private:
    std::weak_ptr<RayCaster> _ray_caster;
    std::shared_ptr<GLActiveTextureCounter> _gl_act_tex_counter;

    // render steps
    std::vector<std::shared_ptr<RCStepBase>> _ray_casting_steps;

    // Ray casting mode cache
    MaskMode _mask_mode;
    CompositeMode _composite_mode;
    InterpolationMode _interpolation_mode;
    ShadingMode _shading_mode;
    ColorInverseMode _color_inverse_mode;
    MaskOverlayMode _mask_overlay_mode;

    // Resource
    GLVAOPtr _gl_vao;
    GLBufferPtr _gl_buffer_vertex;
    GLProgramPtr _gl_program;
    GLTimeQueryPtr _gl_time_query;
    GLResourceShield _res_shield;

    double _render_duration;
    // For Testing
    int _last_test_code;
};

MED_IMG_END_NAMESPACE

#endif