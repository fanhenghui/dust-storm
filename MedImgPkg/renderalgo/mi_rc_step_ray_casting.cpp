#include "mi_rc_step_ray_casting.h"
#include "mi_shader_collection.h"

#include "glresource/mi_gl_program.h"
#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_utils.h"

#include "io/mi_image_data.h"

#include "mi_ray_caster.h"

MED_IMG_BEGIN_NAMESPACE 

void RCStepRayCastingMIPBase::set_gpu_parameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr program = _gl_program.lock();
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();

    //Pseudo color
    unsigned int uiLength(1);
    GLTexture1DPtr pPseudoColorTex = ray_caster->get_pseudo_color_texture(uiLength);
    RENDERALGO_CHECK_NULL_EXCEPTION(pPseudoColorTex);

    glEnable(GL_TEXTURE_1D);
    const int act_tex = _act_tex_counter->tick();
    glActiveTexture(GL_TEXTURE0 + act_tex);
    pPseudoColorTex->bind();
    GLTextureUtils::set_1d_wrap_s(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_1D , GL_LINEAR);
    glUniform1i(_loc_pseudo_color , act_tex);
    glDisable(GL_TEXTURE_1D);

    //Linear change: discrete array (0 ~ length -1) to texture coordinate(0 ~ 1)
    glUniform1f(_loc_pseudo_color_slope,  (uiLength- 1.0f)/uiLength);
    glUniform1f(_loc_pseudo_color_intercept,  0.5f/uiLength);

    //Global window level
    std::shared_ptr<ImageData> pVolume = ray_caster->get_volume_data();
    RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

    float ww(1), wl(0);
    ray_caster->get_global_window_level(ww , wl);
    pVolume->normalize_wl(ww, wl);
    glUniform2f(_loc_global_wl , ww, wl);

    CHECK_GL_ERROR;
}

void RCStepRayCastingMIPBase::get_uniform_location()
{
    GLProgramPtr program = _gl_program.lock();
    _loc_pseudo_color = program->get_uniform_location("pseudo_color");
    _loc_pseudo_color_slope =  program->get_uniform_location("pseudo_color_slope");
    _loc_pseudo_color_intercept = program->get_uniform_location("pseudo_color_intercept");
    _loc_global_wl = program->get_uniform_location("global_wl");

    if (-1 == _loc_pseudo_color ||
        -1 == _loc_pseudo_color_slope ||
        -1 == _loc_pseudo_color_intercept ||
        -1 == _loc_global_wl)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}

GLShaderInfo RCStepRayCastingAverage::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_RAY_CASTING_AVERAGE_FRAG , "RCStepRayCastingAverage");
}


GLShaderInfo RCStepRayCastingMIPMinIP::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_RAY_CASTING_MIP_MINIP_FRAG , "RCStepRayCastingAverage");
}


GLShaderInfo RCStepRayCastingDVR::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_RAY_CASTING_DVR_FRAG , "RCStepRayCastingDVR");
}

MED_IMG_END_NAMESPACE