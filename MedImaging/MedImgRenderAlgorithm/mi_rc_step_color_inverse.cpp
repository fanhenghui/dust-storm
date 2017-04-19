#include "mi_rc_step_color_inverse.h"
#include "mi_shader_collection.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepColorInverseDisable::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCColorInverseDisableFrag , "RCStepColorInverseDisable");
}

GLShaderInfo RCStepColorInverseEnable::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCColorInverseEnableFrag , "RCStepColorInverseEnable");
}

MED_IMAGING_END_NAMESPACE