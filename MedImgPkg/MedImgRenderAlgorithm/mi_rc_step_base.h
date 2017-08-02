#ifndef MED_IMG_RAY_CASTING_GPU_STEP_BASE_H
#define MED_IMG_RAY_CASTING_GPU_STEP_BASE_H

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

class RayCaster;
class RCStepBase
{
public:
    RCStepBase(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program)
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(program);
        _gl_program = program;
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster);
        _ray_caster = ray_caster;
    };

    virtual ~RCStepBase(){};

    virtual GLShaderInfo get_shader_info() = 0;

    void set_active_texture_counter(std::shared_ptr<GLActiveTextureCounter> tex_counter)
    {
        _act_tex_counter = tex_counter;
    }

    virtual void set_gpu_parameter() 
    {
    }

    virtual void get_uniform_location()
    {

    }

protected:
    std::weak_ptr<GLProgram> _gl_program;
    std::weak_ptr<RayCaster> _ray_caster;
    std::shared_ptr<GLActiveTextureCounter> _act_tex_counter;
};


MED_IMG_END_NAMESPACE
#endif