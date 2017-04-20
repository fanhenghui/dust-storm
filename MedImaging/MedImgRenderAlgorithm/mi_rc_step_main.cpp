#include "mi_rc_step_main.h"
#include "mi_shader_collection.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgIO/mi_image_data.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"

MED_IMAGING_BEGIN_NAMESPACE


GLShaderInfo RCStepMainVert::get_shader_info()
{
    
    return GLShaderInfo(GL_VERTEX_SHADER , ksRCMainVert , "RCStepMainVert");
}

void RCStepMainVert::set_gpu_parameter()
{
}

GLShaderInfo RCStepMainFrag::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , ksRCMainFrag , "RCStepMainFrag");
}

void RCStepMainFrag::set_gpu_parameter()
{
    CHECK_GL_ERROR;

    GLProgramPtr pProgram = m_pProgram.lock();
    std::shared_ptr<RayCaster> pRayCaster = m_pRayCaster.lock();
    std::shared_ptr<ImageData> pVolumeData = pRayCaster->get_volume_data();

    RENDERALGO_CHECK_NULL_EXCEPTION(pVolumeData);

    //1 Entry exit points
    std::shared_ptr<EntryExitPoints> pEE = pRayCaster->get_entry_exit_points();
    RENDERALGO_CHECK_NULL_EXCEPTION(pEE);

    GLTexture2DPtr pEntryTex = pEE->get_entry_points_texture();
    GLTexture2DPtr pExitTex = pEE->get_exit_points_texture();


#define IMG_BINDING_ENTRY_POINTS  0
#define IMG_BINDING_EXIT_POINTS  1

    pEntryTex->bind_image(IMG_BINDING_ENTRY_POINTS , 0 , GL_FALSE , 0 , GL_READ_ONLY , GL_RGBA32F);
    pExitTex->bind_image(IMG_BINDING_EXIT_POINTS , 0 , GL_FALSE , 0 , GL_READ_ONLY , GL_RGBA32F);


#undef IMG_BINDING_ENTRY_POINTS
#undef IMG_BINDING_EXIT_POINTS

    //2 Volume texture
    std::vector<GLTexture3DPtr> vecVolumeTex = pRayCaster->get_volume_data_texture();
    if (vecVolumeTex.empty())
    {
        RENDERALGO_THROW_EXCEPTION("Volume texture is empty!");
    }
    glEnable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE1);
    vecVolumeTex[0]->bind();
    GLTextureUtils::set_1d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_LINEAR);
    glUniform1i(m_iLocVolumeData , 1);
    glDisable(GL_TEXTURE_3D);

    //3 Volume dimension
    glUniform3f(m_iLocVolumeDim , (float)pVolumeData->m_uiDim[0] , 
        (float)pVolumeData->m_uiDim[1] , (float)pVolumeData->m_uiDim[2]);

    //4 Sample rate
    glUniform1f(m_iLocSampleRate , pRayCaster->get_sample_rate());

    //TODO Mask related



    CHECK_GL_ERROR;
}

void RCStepMainFrag::get_uniform_location()
{
    GLProgramPtr pProgram = m_pProgram.lock();
    m_iLocVolumeDim = pProgram->get_uniform_location("vVolumeDim");
    m_iLocVolumeData = pProgram->get_uniform_location("sVolume");
    m_iLocMaskData = pProgram->get_uniform_location("sMask");
    m_iLocSampleRate = pProgram->get_uniform_location("fSampleRate");

    if (-1 == m_iLocVolumeDim ||
        -1 == m_iLocVolumeData ||
        //-1 == m_iLocMaskData ||
        -1 == m_iLocSampleRate)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}



MED_IMAGING_END_NAMESPACE