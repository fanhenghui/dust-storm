#ifndef MED_IMAGING_RAY_CASTING_GPU_H_
#define MED_IMAGING_RAY_CASTING_GPU_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"

MED_IMAGING_BEGIN_NAMESPACE
class RayCaster;
class RCStepBase;
class GLActiveTextureCounter;
class RayCastingGPU
{
public:
    RayCastingGPU(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingGPU();

    void render(int test_code = 0);

private:
    void update_i();

private:
    std::weak_ptr<RayCaster> _ray_caster;
    GLProgramPtr _program;
    std::shared_ptr<GLActiveTextureCounter> _gl_act_tex_counter;

    //render steps
    std::vector<std::shared_ptr<RCStepBase>> _ray_casting_steps;

    //Ray casting mode cache
    MaskMode _mask_mode;
    CompositeMode _composite_mode;
    InterpolationMode _interpolation_mode;
    ShadingMode _shading_mode;
    ColorInverseMode _color_inverse_mode;

    //VAO
    GLVAOPtr _gl_vao;
    GLBufferPtr _gl_buffer_vertex;
};

MED_IMAGING_END_NAMESPACE

#endif