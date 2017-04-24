#include "mi_rc_step_color_inverse.h"
#include "mi_shader_collection.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepColorInverseDisable::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_COLOR_INVERSE_DISABLE_FRAG , "RCStepColorInverseDisable");
}

GLShaderInfo RCStepColorInverseEnable::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_COLOR_INVERSE_ENABLE_FRAG , "RCStepColorInverseEnable");
}

MED_IMAGING_END_NAMESPACE