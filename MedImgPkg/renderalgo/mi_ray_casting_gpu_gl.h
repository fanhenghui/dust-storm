#ifndef MEDIMGRENDERALGO_RAY_CASTING_GPU_H
#define MEDIMGRENDERALGO_RAY_CASTING_GPU_H

#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE
class RayCaster;
class RCStepBase;
class GLActiveTextureCounter;
class RayCastingGPUGL {
public:
    RayCastingGPUGL(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingGPUGL();

    float get_rendering_duration() const;

    void render();

private:
    void update_i();

private:
    std::weak_ptr<RayCaster> _ray_caster;
    std::shared_ptr<GLActiveTextureCounter> _active_texture_counter;

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
    GLVAOPtr _vao;
    GLBufferPtr _buffer_vertex;
    GLProgramPtr _program;
    GLTimeQueryPtr _time_query;
    GLResourceShield _res_shield;

    float _render_duration;
    // For Testing
    int _last_test_code;

private:
    DISALLOW_COPY_AND_ASSIGN(RayCastingGPUGL);
};

MED_IMG_END_NAMESPACE

#endif