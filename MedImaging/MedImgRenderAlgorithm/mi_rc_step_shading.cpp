#include "mi_rc_step_shading.h"
#include "mi_shader_collection.h"

MED_IMAGING_BEGIN_NAMESPACE


GLShaderInfo RCStepShadingNone::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCShadingNoneFrag , "RCStepShadingNone");
}

GLShaderInfo RCStepShadingPhong::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCShadingPhongFrag , "RCStepShadingPhong");
}

void RCStepShadingPhong::SetGPUParameter()
{

}

MED_IMAGING_END_NAMESPACE