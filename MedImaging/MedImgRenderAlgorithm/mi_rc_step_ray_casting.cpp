#include "mi_rc_step_ray_casting.h"
#include "mi_shader_collection.h"

#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"

MED_IMAGING_BEGIN_NAMESPACE

void RCStepRayCastingMIPBase::SetGPUParameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr pProgram = m_pProgram.lock();
    std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();

    //Pseudo color
    unsigned int uiLength(1);
    GLTexture1DPtr pPseudoColorTex = pRayCaster->GetPseudoColorTexture(uiLength);
    RENDERALGO_CHECK_NULL_EXCEPTION(pPseudoColorTex);

    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE2);
    pPseudoColorTex->Bind();
    GLTextureUtils::Set1DWrapS(GL_CLAMP_TO_BORDER);
    GLTextureUtils::SetFilter(GL_TEXTURE_1D , GL_LINEAR);
    glUniform1i(m_iLocPseudoColor , 2);
    glDisable(GL_TEXTURE_1D);

    glUniform1f(m_iLocPseudoColorSlope,  (uiLength- 1.0f)/uiLength);
    glUniform1f(m_iLocPseudoColorIntercept,  0.5f/uiLength);

    //Global window level
    std::shared_ptr<ImageData> pVolume = pRayCaster->GetVolumeData();
    RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

    float fWW(1), fWL(0);
    pRayCaster->GetGlobalWindowLevel(fWW , fWL);
    pVolume->NormalizeWindowLevel(fWW, fWL);
    glUniform2f(m_iLocGlobalWL , fWW, fWL);

    CHECK_GL_ERROR;
}

void RCStepRayCastingMIPBase::GetUniformLocation()
{
    GLProgramPtr pProgram = m_pProgram.lock();
    m_iLocPseudoColor = pProgram->GetUniformLocation("sPseudoColor");
    m_iLocPseudoColorSlope =  pProgram->GetUniformLocation("fPseudoColorSlope");
    m_iLocPseudoColorIntercept = pProgram->GetUniformLocation("fPseudoColorIntercept");
    m_iLocGlobalWL = pProgram->GetUniformLocation("vGlobalWL");

    if (-1 == m_iLocPseudoColor ||
        -1 == m_iLocPseudoColorSlope ||
        -1 == m_iLocPseudoColorIntercept ||
        -1 == m_iLocGlobalWL)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}

GLShaderInfo RCStepRayCastingAverage::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCRayCastingAverageFrag , "RCStepRayCastingAverage");
}


GLShaderInfo RCStepRayCastingMIPMinIP::GetShaderInfo()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCRayCastingMIPMinIPFrag , "RCStepRayCastingAverage");
}


MED_IMAGING_END_NAMESPACE