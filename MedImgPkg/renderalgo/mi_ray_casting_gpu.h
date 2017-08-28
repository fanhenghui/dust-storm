#ifndef MED_IMG_RAY_CASTING_GPU_H_
#define MED_IMG_RAY_CASTING_GPU_H_

#include "renderalgo/mi_render_algo_export.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE 
class RayCaster;
class RCStepBase;
class GLActiveTextureCounter;
class RayCastingGPU
{
public:
    RayCastingGPU(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingGPU();

    void render();

private:
    void update_i();

private:
    std::weak_ptr<RayCaster> _ray_caster;
    std::shared_ptr<GLActiveTextureCounter> _gl_act_tex_counter;

    //render steps
    std::vector<std::shared_ptr<RCStepBase>> _ray_casting_steps;

    //Ray casting mode cache
    MaskMode _mask_mode;
    CompositeMode _composite_mode;
    InterpolationMode _interpolation_mode;
    ShadingMode _shading_mode;
    ColorInverseMode _color_inverse_mode;
    MaskOverlayMode _mask_overlay_mode;

    //Resource
    GLVAOPtr _gl_vao;
    GLBufferPtr _gl_buffer_vertex;
    GLProgramPtr _gl_program;
    GLResourceShield _res_shield;

    //For Testing
    int _last_test_code;
};

MED_IMG_END_NAMESPACE

#endif