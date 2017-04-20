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
    RCStepBase(std::shared_ptr<RayCaster> pRayCaster , std::shared_ptr<GLProgram>  pProgram)
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(pProgram);
        m_pProgram = pProgram;
        RENDERALGO_CHECK_NULL_EXCEPTION(pRayCaster);
        m_pRayCaster = pRayCaster;
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
    std::weak_ptr<GLProgram> m_pProgram;
    std::weak_ptr<RayCaster> m_pRayCaster;
};


MED_IMAGING_END_NAMESPACE
#endif