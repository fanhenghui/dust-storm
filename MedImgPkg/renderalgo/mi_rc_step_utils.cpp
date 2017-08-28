#include "mi_rc_step_utils.h"
#include "mi_shader_collection.h"

MED_IMG_BEGIN_NAMESPACE 

GLShaderInfo RCStepUtils::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCUtilsFrag , "RCStepUtils");
}

MED_IMG_END_NAMESPACE