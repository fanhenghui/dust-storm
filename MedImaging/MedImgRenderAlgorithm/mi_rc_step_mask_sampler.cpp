#include "mi_rc_step_mask_sampler.h"
#include "mi_shader_collection.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepMaskNoneSampler::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCMaskNoneSamplerFrag , "RCStepMaskNoneSampler");
}


GLShaderInfo RCStepMaskNearstSampler::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCMaskNearstSamplerFrag , "RCStepMaskNearstSampler");
}

MED_IMAGING_END_NAMESPACE