#include "mi_rc_step_composite.h"
#include "mi_shader_collection.h"

#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepCompositeAverage::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeAverageFrag , "RCStepCompositeAverage");
}


GLShaderInfo RCStepCompositeMIP::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeMIPFrag , "RCStepCompositeMIP");
}


GLShaderInfo RCStepCompositeMinIP::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeMinIPFrag , "RCStepCompositeMinIP");
}

void RCStepCompositeMinIP::set_gpu_parameter()
{

}

void RCStepCompositeMinIP::get_uniform_location()
{
    GLProgramPtr pProgram = m_pProgram.lock();
    m_iLocCustomMinThreshold = pProgram->get_uniform_location("fCustomMinThreshold");

    if (-1 == m_iLocCustomMinThreshold)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}


GLShaderInfo RCStepCompositeDVR::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeDVRFrag , "RCStepCompositeDVR");
}

void RCStepCompositeDVR::set_gpu_parameter()
{
    //TODO
}

void RCStepCompositeDVR::get_uniform_location()
{
    //TODO
}

MED_IMAGING_END_NAMESPACE