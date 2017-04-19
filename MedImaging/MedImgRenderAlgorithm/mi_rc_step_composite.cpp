#include "mi_rc_step_composite.h"
#include "mi_shader_collection.h"

#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepCompositeAverage::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeAverageFrag , "RCStepCompositeAverage");
}


GLShaderInfo RCStepCompositeMIP::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeMIPFrag , "RCStepCompositeMIP");
}


GLShaderInfo RCStepCompositeMinIP::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeMinIPFrag , "RCStepCompositeMinIP");
}

void RCStepCompositeMinIP::SetGPUParameter()
{

}

void RCStepCompositeMinIP::GetUniformLocation()
{
    GLProgramPtr pProgram = m_pProgram.lock();
    m_iLocCustomMinThreshold = pProgram->GetUniformLocation("fCustomMinThreshold");

    if (-1 == m_iLocCustomMinThreshold)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}


GLShaderInfo RCStepCompositeDVR::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeDVRFrag , "RCStepCompositeDVR");
}

void RCStepCompositeDVR::SetGPUParameter()
{
    //TODO
}

void RCStepCompositeDVR::GetUniformLocation()
{
    //TODO
}

MED_IMAGING_END_NAMESPACE