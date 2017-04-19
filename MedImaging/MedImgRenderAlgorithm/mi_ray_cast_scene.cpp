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


    if (CPU == Configuration::Instance()->GetProcessingUnitType())
    {
        m_pRayCaster->SetStrategy(CPU_BASE);
    }
    else
    {
        m_pRayCaster->SetStrategy(GPU_BASE);
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


    if (CPU == Configuration::Instance()->GetProcessingUnitType())
    {
        m_pRayCaster->SetStrategy(CPU_BASE);
    }
    else
    {
        m_pRayCaster->SetStrategy(GPU_BASE);
    }
}

RayCastScene::~RayCastScene()
{

}

void RayCastScene::Initialize()
{
    SceneBase::Initialize();

    //Canvas
    m_pCanvas->SetDisplaySize(m_iWidth , m_iHeight);
    m_pCanvas->Initialize();
}

void RayCastScene::Finalize()
{
    m_pCanvas->Finialize();
    m_pEntryExitPoints->Finialize();
    m_pRayCaster->Finialize();
}

void RayCastScene::SetDisplaySize(int iWidth , int iHeight)
{
    SceneBase::SetDisplaySize(iWidth ,iHeight);
    m_pCanvas->SetDisplaySize(iWidth , iHeight);
    m_pCanvas->UpdateFBO();//Update texture size
    m_pEntryExitPoints->SetDisplaySize(iWidth , iHeight);
    m_pCameraInteractor->Resize(iWidth , iHeight);
}

void RayCastScene::Render(int iTestCode)
{
    //Skip render scene
    if (!GetDirty())
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

    m_pEntryExitPoints->CalculateEntryExitPoints();
    m_pRayCaster->Render(m_iTestCode);

    glPopAttrib();
    CHECK_GL_ERROR;

    //////////////////////////////////////////////////////////////////////////
    //2 Mapping ray casting result to Scene FBO
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,m_iWidth , m_iHeight);

    m_pSceneFBO->Bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    CHECK_GL_ERROR;

    glEnable(GL_TEXTURE_2D);
    m_pCanvas->GetColorAttachTexture()->Bind();

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

    SetDirty(false);
}

void RayCastScene::SetVolumeInfos(std::shared_ptr<VolumeInfos> pVolumeInfos)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolumeInfos);
        m_pVolumeInfos = pVolumeInfos;

        std::shared_ptr<ImageData> pVolume = m_pVolumeInfos->GetVolume();
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

        std::shared_ptr<ImageDataHeader> pDataHeader = m_pVolumeInfos->GetDataHeader();
        RENDERALGO_CHECK_NULL_EXCEPTION(pDataHeader);

        //Camera calculator
        m_pCameraCalculator = pVolumeInfos->GetCameraCalculator();

        //Entry exit points
        m_pEntryExitPoints->SetImageData(pVolume);
        m_pEntryExitPoints->SetCamera(m_pCamera);
        m_pEntryExitPoints->SetDisplaySize(m_iWidth , m_iHeight);
        m_pEntryExitPoints->SetCameraCalculator(m_pCameraCalculator);
        m_pEntryExitPoints->Initialize();

        //Ray caster
        m_pRayCaster->SetCanvas(m_pCanvas);
        m_pRayCaster->SetEntryExitPoints(m_pEntryExitPoints);
        m_pRayCaster->SetCamera(m_pCamera);
        m_pRayCaster->SetVolumeToWorldMatrix(m_pCameraCalculator->GetVolumeToWorldMatrix());
        m_pRayCaster->SetVolumeData(pVolume);
        //m_pRayCaster->SetMaskData(pMask);

        if (GPU == Configuration::Instance()->GetProcessingUnitType())
        {
            //Initialize gray pseudo color texture
            if (!m_pPseudoColor)
            {
                UIDType uid;
                m_pPseudoColor = GLResourceManagerContainer::Instance()->GetTexture1DManager()->CreateObject(uid);
                m_pPseudoColor->SetDescription("Pseudo color texture gray");
                m_pPseudoColor->Initialize();
                m_pPseudoColor->Bind();
                GLTextureUtils::Set1DWrapS(GL_CLAMP_TO_EDGE);
                GLTextureUtils::SetFilter(GL_TEXTURE_1D , GL_LINEAR);
                unsigned char pData[] = {0,0,0,0,255,255,255,255};
                m_pPseudoColor->Load(GL_RGBA8 , 2, GL_RGBA , GL_UNSIGNED_BYTE , pData);
            }
            m_pRayCaster->SetPseudoColorTexture(m_pPseudoColor , 2);

            //Volume texture
            m_pRayCaster->SetVolumeDataTexture(pVolumeInfos->GetVolumeTexture());
        }
        m_pRayCaster->Initialize();

        SetDirty(true);
    }
    catch (const Exception& e)
    {
        //TOOD LOG
        std::cout << e.what();
        //assert(false);
        throw e;
    }
}

void RayCastScene::SetMaskLabelLevel(LabelLevel eLabelLevel)
{
    m_pRayCaster->SetMaskLabelLevel(eLabelLevel);
    SetDirty(true);
}

void RayCastScene::SetSampleRate(float fSampleRate)
{
    m_pRayCaster->SetSampleRate(fSampleRate);
    SetDirty(true);
}

void RayCastScene::SetVisibleLabels(std::vector<unsigned char> vecLabels)
{
    m_pRayCaster->SetVisibleLabels(vecLabels);
    SetDirty(true);
}

void RayCastScene::SetWindowlevel(float fWW , float fWL , unsigned char ucLabel)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolumeInfos);
    m_pVolumeInfos->GetVolume()->RegulateWindowLevel(fWW , fWL);

    m_pRayCaster->SetWindowlevel(fWW , fWL , ucLabel);

    SetDirty(true);
}

void RayCastScene::SetGlobalWindowLevel(float fWW , float fWL)
{
    m_fGlobalWW = fWW;
    m_fGlobalWL = fWL;

    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolumeInfos);
    m_pVolumeInfos->GetVolume()->RegulateWindowLevel(fWW , fWL);

    m_pRayCaster->SetGlobalWindowLevel(fWW, fWL);

    SetDirty(true);
}

void RayCastScene::SetMaskMode(MaskMode eMode)
{
    m_pRayCaster->SetMaskMode(eMode);

    SetDirty(true);
}

void RayCastScene::SetCompositeMode(CompositeMode eMode)
{
    m_pRayCaster->SetCompositeMode(eMode);

    SetDirty(true);
}

void RayCastScene::SetInterpolationMode(InterpolationMode eMode)
{
    m_pRayCaster->SetInterpolationMode(eMode);

    SetDirty(true);
}

void RayCastScene::SetShadingMode(ShadingMode eMode)
{
    m_pRayCaster->SetShadingMode(eMode);

    SetDirty(true);
}

void RayCastScene::SetColorInverseMode(ColorInverseMode eMode)
{
    m_pRayCaster->SetColorInverseMode(eMode);

    SetDirty(true);
}

void RayCastScene::SetTestCode(int iTestCode)
{
    m_iTestCode = iTestCode;

    SetDirty(true);
}

void RayCastScene::GetGlobalWindowLevel(float& fWW , float& fWL) const
{
    fWW = m_fGlobalWW;
    fWL = m_fGlobalWL;
}

std::shared_ptr<VolumeInfos> RayCastScene::GetVolumeInfos() const
{
    return m_pVolumeInfos;
}

std::shared_ptr<CameraCalculator> RayCastScene::GetCameraCalculator() const
{
    return m_pCameraCalculator;
}



MED_IMAGING_END_NAMESPACE