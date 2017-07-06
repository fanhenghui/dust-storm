#include "mi_rc_step_mask_sampler.h"
#include "mi_shader_collection.h"

MED_IMG_BEGIN_NAMESPACE

GLShaderInfo RCStepMaskNoneSampler::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_MASK_NONE_SAMPLER_FRAG , "RCStepMaskNoneSampler");
}


GLShaderInfo RCStepMaskNearstSampler::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_MASK_NEARST_SAMPLER_FRAG , "RCStepMaskNearstSampler");
}

MED_IMG_END_NAMESPACE