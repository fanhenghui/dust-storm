#include "mi_rc_step_shading.h"
#include "mi_shader_collection.h"

MED_IMG_BEGIN_NAMESPACE


GLShaderInfo RCStepShadingNone::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_SHADING_NONE_FRAG , "RCStepShadingNone");
}

GLShaderInfo RCStepShadingPhong::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_SHADING_PHONG_FRAG , "RCStepShadingPhong");
}

void RCStepShadingPhong::set_gpu_parameter()
{

}

MED_IMG_END_NAMESPACE