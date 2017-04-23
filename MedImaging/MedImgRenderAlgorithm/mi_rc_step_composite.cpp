#include "mi_rc_step_composite.h"
#include "mi_shader_collection.h"

#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"

MED_IMAGING_BEGIN_NAMESPACE

GLShaderInfo RCStepCompositeAverage::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeAverageFrag , "RCStepCompositeAverage");
}


GLShaderInfo RCStepCompositeMIP::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeMIPFrag , "RCStepCompositeMIP");
}


GLShaderInfo RCStepCompositeMinIP::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeMinIPFrag , "RCStepCompositeMinIP");
}

void RCStepCompositeMinIP::set_gpu_parameter()
{

}

void RCStepCompositeMinIP::get_uniform_location()
{
    GLProgramPtr program = _program.lock();
    _loc_custom_min_threshold = program->get_uniform_location("custom_min_threshold");

    if (-1 == _loc_custom_min_threshold)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}


GLShaderInfo RCStepCompositeDVR::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCCompositeDVRFrag , "RCStepCompositeDVR");
}

void RCStepCompositeDVR::set_gpu_parameter()
{
    //TODO
}

void RCStepCompositeDVR::get_uniform_location()
{
    //TODO
}

MED_IMAGING_END_NAMESPACE