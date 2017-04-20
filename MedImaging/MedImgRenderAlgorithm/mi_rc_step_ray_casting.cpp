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

    GLProgramPtr pProgram = m_pProgram.lock();
    std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();

    //Pseudo color
    unsigned int uiLength(1);
    GLTexture1DPtr pPseudoColorTex = pRayCaster->get_pseudo_color_texture(uiLength);
    RENDERALGO_CHECK_NULL_EXCEPTION(pPseudoColorTex);

    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE2);
    pPseudoColorTex->bind();
    GLTextureUtils::set_1d_wrap_s(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_1D , GL_LINEAR);
    glUniform1i(m_iLocPseudoColor , 2);
    glDisable(GL_TEXTURE_1D);

    glUniform1f(m_iLocPseudoColorSlope,  (uiLength- 1.0f)/uiLength);
    glUniform1f(m_iLocPseudoColorIntercept,  0.5f/uiLength);

    //Global window level
    std::shared_ptr<ImageData> pVolume = pRayCaster->get_volume_data();
    RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

    float fWW(1), fWL(0);
    pRayCaster->get_global_window_level(fWW , fWL);
    pVolume->normalize_wl(fWW, fWL);
    glUniform2f(m_iLocGlobalWL , fWW, fWL);

    CHECK_GL_ERROR;
}

void RCStepRayCastingMIPBase::get_uniform_location()
{
    GLProgramPtr pProgram = m_pProgram.lock();
    m_iLocPseudoColor = pProgram->get_uniform_location("sPseudoColor");
    m_iLocPseudoColorSlope =  pProgram->get_uniform_location("fPseudoColorSlope");
    m_iLocPseudoColorIntercept = pProgram->get_uniform_location("fPseudoColorIntercept");
    m_iLocGlobalWL = pProgram->get_uniform_location("vGlobalWL");

    if (-1 == m_iLocPseudoColor ||
        -1 == m_iLocPseudoColorSlope ||
        -1 == m_iLocPseudoColorIntercept ||
        -1 == m_iLocGlobalWL)
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