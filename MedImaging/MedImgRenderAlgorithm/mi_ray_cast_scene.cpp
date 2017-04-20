#include "mi_ray_cast_scene.h"


#include "MedImgCommon/mi_configuration.h"

#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_point2.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"


MED_IMAGING_BEGIN_NAMESPACE

RayCastScene::RayCastScene():SceneBase(),m_iTestCode(0),
    m_fGlobalWW(0),
    m_fGlobalWL(0)
{
    m_pRayCastCamera.reset(new OrthoCamera());
    m_pCamera = m_pRayCastCamera;

    m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pRayCastCamera));

    m_pRayCaster.reset(new RayCaster());

    m_pCanvas.reset(new RayCasterCanvas());


    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        m_pRayCaster->set_strategy(CPU_BASE);
    }
    else
    {
        m_pRayCaster->set_strategy(GPU_BASE);
    }
}

RayCastScene::RayCastScene(int iWidth , int iHeight):SceneBase(iWidth , iHeight),m_iTestCode(0),
    m_fGlobalWW(0),
    m_fGlobalWL(0)
{
    m_pRayCastCamera.reset(new OrthoCamera());
    m_pCamera = m_pRayCastCamera;

    m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pRayCastCamera));

    m_pRayCaster.reset(new RayCaster());

    m_pCanvas.reset(new RayCasterCanvas());


    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        m_pRayCaster->set_strategy(CPU_BASE);
    }
    else
    {
        m_pRayCaster->set_strategy(GPU_BASE);
    }
}

RayCastScene::~RayCastScene()
{

}

void RayCastScene::initialize()
{
    SceneBase::initialize();

    //Canvas
    m_pCanvas->set_display_size(m_iWidth , m_iHeight);
    m_pCanvas->initialize();
}

void RayCastScene::finalize()
{
    m_pCanvas->finialize();
    m_pEntryExitPoints->finialize();
    m_pRayCaster->finialize();
}

void RayCastScene::set_display_size(int iWidth , int iHeight)
{
    SceneBase::set_display_size(iWidth ,iHeight);
    m_pCanvas->set_display_size(iWidth , iHeight);
    m_pCanvas->update_fbo();//update texture size
    m_pEntryExitPoints->set_display_size(iWidth , iHeight);
    m_pCameraInteractor->resize(iWidth , iHeight);
}

void RayCastScene::render(int iTestCode)
{
    //Skip render scene
    if (!get_dirty())
    {
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    //TODO other common graphic object rendering list

    //////////////////////////////////////////////////////////////////////////
    //1 Ray casting
    CHECK_GL_ERROR;
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,m_iWidth , m_iHeight);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    m_pEntryExitPoints->calculate_entry_exit_points();
    m_pRayCaster->render(m_iTestCode);

    glPopAttrib();
    CHECK_GL_ERROR;

    //////////////////////////////////////////////////////////////////////////
    //2 Mapping ray casting result to Scene FBO
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,m_iWidth , m_iHeight);

    m_pSceneFBO->bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    CHECK_GL_ERROR;

    glEnable(GL_TEXTURE_2D);
    m_pCanvas->get_color_attach_texture()->bind();

    CHECK_GL_ERROR;

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); 
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); 
    glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); 
    glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0);
    glVertex2f(-1.0, 1.0);
    glEnd();

    //glDisable(GL_TEXTURE_2D);
    CHECK_GL_ERROR;
    glPopAttrib();//TODO Here will give a GL_INVALID_OPERATION error !!!
    CHECK_GL_ERROR;

    set_dirty(false);
}

void RayCastScene::set_volume_infos(std::shared_ptr<VolumeInfos> pVolumeInfos)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolumeInfos);
        m_pVolumeInfos = pVolumeInfos;

        std::shared_ptr<ImageData> pVolume = m_pVolumeInfos->get_volume();
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

        std::shared_ptr<ImageDataHeader> pDataHeader = m_pVolumeInfos->get_data_header();
        RENDERALGO_CHECK_NULL_EXCEPTION(pDataHeader);

        //Camera calculator
        m_pCameraCalculator = pVolumeInfos->get_camera_calculator();

        //Entry exit points
        m_pEntryExitPoints->set_image_data(pVolume);
        m_pEntryExitPoints->set_camera(m_pCamera);
        m_pEntryExitPoints->set_display_size(m_iWidth , m_iHeight);
        m_pEntryExitPoints->set_camera_calculator(m_pCameraCalculator);
        m_pEntryExitPoints->initialize();

        //Ray caster
        m_pRayCaster->set_canvas(m_pCanvas);
        m_pRayCaster->set_entry_exit_points(m_pEntryExitPoints);
        m_pRayCaster->set_camera(m_pCamera);
        m_pRayCaster->set_volume_to_world_matrix(m_pCameraCalculator->get_volume_to_world_matrix());
        m_pRayCaster->set_volume_data(pVolume);
        //m_pRayCaster->set_mask_data(pMask);

        if (GPU == Configuration::instance()->get_processing_unit_type())
        {
            //initialize gray pseudo color texture
            if (!m_pPseudoColor)
            {
                UIDType uid;
                m_pPseudoColor = GLResourceManagerContainer::instance()->get_texture_1d_manager()->create_object(uid);
                m_pPseudoColor->set_description("Pseudo color texture gray");
                m_pPseudoColor->initialize();
                m_pPseudoColor->bind();
                GLTextureUtils::set_1d_wrap_s(GL_CLAMP_TO_EDGE);
                GLTextureUtils::set_filter(GL_TEXTURE_1D , GL_LINEAR);
                unsigned char pData[] = {0,0,0,0,255,255,255,255};
                m_pPseudoColor->load(GL_RGBA8 , 2, GL_RGBA , GL_UNSIGNED_BYTE , pData);
            }
            m_pRayCaster->set_pseudo_color_texture(m_pPseudoColor , 2);

            //Volume texture
            m_pRayCaster->set_volume_data_texture(pVolumeInfos->get_volume_texture());
        }
        m_pRayCaster->initialize();

        set_dirty(true);
    }
    catch (const Exception& e)
    {
        //TOOD LOG
        std::cout << e.what();
        //assert(false);
        throw e;
    }
}

void RayCastScene::set_mask_label_level(LabelLevel eLabelLevel)
{
    m_pRayCaster->set_mask_label_level(eLabelLevel);
    set_dirty(true);
}

void RayCastScene::set_sample_rate(float fSampleRate)
{
    m_pRayCaster->set_sample_rate(fSampleRate);
    set_dirty(true);
}

void RayCastScene::set_visible_labels(std::vector<unsigned char> vecLabels)
{
    m_pRayCaster->set_visible_labels(vecLabels);
    set_dirty(true);
}

void RayCastScene::set_window_level(float fWW , float fWL , unsigned char ucLabel)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolumeInfos);
    m_pVolumeInfos->get_volume()->regulate_wl(fWW , fWL);

    m_pRayCaster->set_window_level(fWW , fWL , ucLabel);

    set_dirty(true);
}

void RayCastScene::set_global_window_level(float fWW , float fWL)
{
    m_fGlobalWW = fWW;
    m_fGlobalWL = fWL;

    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolumeInfos);
    m_pVolumeInfos->get_volume()->regulate_wl(fWW , fWL);

    m_pRayCaster->set_global_window_level(fWW, fWL);

    set_dirty(true);
}

void RayCastScene::set_mask_mode(MaskMode eMode)
{
    m_pRayCaster->set_mask_mode(eMode);

    set_dirty(true);
}

void RayCastScene::set_composite_mode(CompositeMode eMode)
{
    m_pRayCaster->set_composite_mode(eMode);

    set_dirty(true);
}

void RayCastScene::set_interpolation_mode(InterpolationMode eMode)
{
    m_pRayCaster->set_interpolation_mode(eMode);

    set_dirty(true);
}

void RayCastScene::set_shading_mode(ShadingMode eMode)
{
    m_pRayCaster->set_shading_mode(eMode);

    set_dirty(true);
}

void RayCastScene::set_color_inverse_mode(ColorInverseMode eMode)
{
    m_pRayCaster->set_color_inverse_mode(eMode);

    set_dirty(true);
}

void RayCastScene::set_test_code(int iTestCode)
{
    m_iTestCode = iTestCode;

    set_dirty(true);
}

void RayCastScene::get_global_window_level(float& fWW , float& fWL) const
{
    fWW = m_fGlobalWW;
    fWL = m_fGlobalWL;
}

std::shared_ptr<VolumeInfos> RayCastScene::get_volume_infos() const
{
    return m_pVolumeInfos;
}

std::shared_ptr<CameraCalculator> RayCastScene::get_camera_calculator() const
{
    return m_pCameraCalculator;
}



MED_IMAGING_END_NAMESPACE