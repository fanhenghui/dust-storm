#ifndef MED_IMAGING_RAY_CASTING_GPU_STEP_BASE_H
#define MED_IMAGING_RAY_CASTING_GPU_STEP_BASE_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_resource_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RayCaster;
class RCStepBase
{
public:
    RCStepBase(std::shared_ptr<RayCaster> ray_caster , std::shared_ptr<GLProgram>  program)
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(program);
        _program = program;
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster);
        _ray_caster = ray_caster;
    };

    virtual ~RCStepBase(){};

    virtual GLShaderInfo get_shader_info() = 0;

    virtual void set_gpu_parameter() 
    {
    }

    virtual void get_uniform_location()
    {

    }

protected:
    std::weak_ptr<GLProgram> _program;
    std::weak_ptr<RayCaster> _ray_caster;
};


MED_IMAGING_END_NAMESPACE
#endif