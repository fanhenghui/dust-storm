#include "mi_rc_step_volume_sampler.h"
#include "mi_shader_collection.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepVolumeNearstSampler::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCVolumeNearstSamplerFrag , "RCStepVolumeNearstSampler");
}


GLShaderInfo RCStepVolumeLinearSampler::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCVolumeLinearSamplerFrag, "RCStepVolumeLinearSampler");
}

MED_IMAGING_END_NAMESPACE