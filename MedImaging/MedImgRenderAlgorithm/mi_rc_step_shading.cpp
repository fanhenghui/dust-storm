#include "mi_rc_step_shading.h"
#include "mi_shader_collection.h"

MED_IMAGING_BEGIN_NAMESPACE


GLShaderInfo RCStepShadingNone::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCShadingNoneFrag , "RCStepShadingNone");
}

GLShaderInfo RCStepShadingPhong::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCShadingPhongFrag , "RCStepShadingPhong");
}

void RCStepShadingPhong::set_gpu_parameter()
{

}

MED_IMAGING_END_NAMESPACE