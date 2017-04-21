#include "mi_rc_step_ray_casting.h"
#include "mi_shader_collection.h"

#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"

MED_IMAGING_BEGIN_NAMESPACE

void RCStepRayCastingMIPBase::set_gpu_parameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr program = _program.lock();
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();

    //Pseudo color
    unsigned int uiLength(1);
    GLTexture1DPtr pPseudoColorTex = ray_caster->get_pseudo_color_texture(uiLength);
    RENDERALGO_CHECK_NULL_EXCEPTION(pPseudoColorTex);

    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE2);
    pPseudoColorTex->bind();
    GLTextureUtils::set_1d_wrap_s(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_1D , GL_LINEAR);
    glUniform1i(_loc_pseudo_color , 2);
    glDisable(GL_TEXTURE_1D);

    glUniform1f(_loc_pseudo_color_slope,  (uiLength- 1.0f)/uiLength);
    glUniform1f(_loc_pseudo_color_intercept,  0.5f/uiLength);

    //Global window level
    std::shared_ptr<ImageData> pVolume = ray_caster->get_volume_data();
    RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

    float fWW(1), fWL(0);
    ray_caster->get_global_window_level(fWW , fWL);
    pVolume->normalize_wl(fWW, fWL);
    glUniform2f(_loc_global_wl , fWW, fWL);

    CHECK_GL_ERROR;
}

void RCStepRayCastingMIPBase::get_uniform_location()
{
    GLProgramPtr program = _program.lock();
    _loc_pseudo_color = program->get_uniform_location("sPseudoColor");
    _loc_pseudo_color_slope =  program->get_uniform_location("fPseudoColorSlope");
    _loc_pseudo_color_intercept = program->get_uniform_location("fPseudoColorIntercept");
    _loc_global_wl = program->get_uniform_location("vGlobalWL");

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
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCRayCastingAverageFrag , "RCStepRayCastingAverage");
}


GLShaderInfo RCStepRayCastingMIPMinIP::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCRayCastingMIPMinIPFrag , "RCStepRayCastingAverage");
}


MED_IMAGING_END_NAMESPACE