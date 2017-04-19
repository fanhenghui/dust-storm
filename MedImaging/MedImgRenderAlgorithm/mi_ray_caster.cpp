#include "mi_ray_caster.h"

#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_texture_1d_array.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster_inner_buffer.h"
#include "mi_ray_casting_cpu.h"
#include "mi_ray_casting_gpu.h"
#include "mi_ray_casting_cpu_brick_acc.h"
#include "mi_ray_caster_canvas.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCaster::RayCaster():m_pInnerBuffer(new RayCasterInnerBuffer()),
m_matVolume2World(Matrix4::kIdentityMatrix),
m_fSampleRate(0.5f),
m_fGlobalWW(1.0f),
m_fGlobalWL(0.0f),
m_pPseudoColorArray(nullptr),
m_uiPseudoColorLength(256),
m_fSSDGray(1.0f),
m_bJitteringEnabled(false),
m_vBoundingMin(Vector3f(0,0,0)),
m_vBoundingMax(Vector3f(32,32,32)),
m_eMaskMode(MASK_NONE),
m_eCompositeMode(COMPOSITE_AVERAGE),
m_eInterpolationMode(LINEAR),
m_eShadingMode(SHADING_NONE),
m_eColorInverseMode(COLOR_INVERSE_DISABLE),
m_eStrategy(CPU_BASE),
m_pBrickCorner(nullptr),
m_pVolumeBrickUnit(nullptr),
m_pMaskBrickUnit(nullptr),
m_pVolumeBrickInfo(nullptr),
m_pMaskBrickInfo(nullptr),
m_uiBrickSize(32),
m_uiBrickExpand(2)
{

}

void RayCaster::Initialize()
{
    
}

void RayCaster::Finialize()
{
    m_pInnerBuffer->ReleaseBuffer();
}

RayCaster::~RayCaster()
{

}

void RayCaster::Render(int iTestCode)
{
    //clock_t t0 = clock();

    if (CPU_BASE == m_eStrategy)
    {
        if (!m_pRayCastingCPU)
        {
            m_pRayCastingCPU.reset(new RayCastingCPU(shared_from_this()));
        }
        m_pRayCastingCPU->Render(iTestCode);
    }
    else if (CPU_BRICK_ACCELERATE == m_eStrategy)
    {
        if (!m_pRayCastingCPUBrickAcc)
        {
            m_pRayCastingCPUBrickAcc.reset(new RayCastingCPUBrickAcc(shared_from_this()));
        }
        m_pRayCastingCPUBrickAcc->Render(iTestCode);
    }
    else if (GPU_BASE == m_eStrategy)
    {
        if (!m_pRayCastingGPU)
        {
            m_pRayCastingGPU.reset(new RayCastingGPU(shared_from_this()));
        }

        {
            CHECK_GL_ERROR;

            FBOStack fboStack;
            m_pCanvas->GetFBO()->Bind();

            CHECK_GL_ERROR;

            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            CHECK_GL_ERROR;

            //Clear 
            glClearColor(0,0,0,0);
            glClearDepth(1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //Viewport
            int iWidth , iHeight;
            m_pCanvas->GetDisplaySize(iWidth , iHeight);
            glViewport(0,0,iWidth , iHeight);

            CHECK_GL_ERROR;

            m_pRayCastingGPU->Render(iTestCode);
        }
    }

    //clock_t t1 = clock();
    //std::cout << "<<<>>><<<>>><<<>>><<<>>><<<>>>\n";
    //std::cout << "Ray casting cost : " << double(t1-t0) << "ms\n";
    //std::cout << "<<<>>><<<>>><<<>>><<<>>><<<>>>\n";
}

void RayCaster::SetVolumeData(std::shared_ptr<ImageData> pImgData)
{
    m_pVolumeData = pImgData;
}

void RayCaster::SetMaskData(std::shared_ptr<ImageData> pImgData)
{
    m_pMaskData = pImgData;
}

void RayCaster::SetVolumeDataTexture(std::vector<GLTexture3DPtr> vecTex)
{
    m_vecVolumeDataTex = vecTex;
}

void RayCaster::SetMaskDataTexture(std::vector<GLTexture3DPtr> vecTex)
{
    m_vecMaskDataTex =vecTex;
}

void RayCaster::SetEntryExitPoints(std::shared_ptr<EntryExitPoints> pEEPs)
{
    m_pEntryExitPoints = pEEPs;
}

void RayCaster::SetCamera(std::shared_ptr<CameraBase> pCamera)
{
    m_pCamera = pCamera;
}

void RayCaster::SetVolumeToWorldMatrix(const Matrix4& mat)
{
    m_matVolume2World = mat;
}

void RayCaster::SetSampleRate(float fSampleRate)
{
    m_fSampleRate = fSampleRate;
}

void RayCaster::SetMaskLabelLevel(LabelLevel eLabelLevel)
{
    m_pInnerBuffer->SetMaskLabelLevel(eLabelLevel);
}

void RayCaster::SetVisibleLabels(std::vector<unsigned char> vecLabels)
{
    m_pInnerBuffer->SetVisibleLabels(vecLabels);
}

void RayCaster::SetWindowlevel(float fWW , float fWL , unsigned char ucLabel)
{
    m_pInnerBuffer->SetWindowLevel(fWW , fWL , ucLabel);
}

void RayCaster::SetGlobalWindowLevel(float fWW , float fWL)
{
    m_fGlobalWW = fWW;
    m_fGlobalWL = fWL;
}

void RayCaster::SetPseudoColorTexture(GLTexture1DPtr pTex , unsigned int uiLength)
{
    m_pPseudoColorTexture = pTex;
    m_uiPseudoColorLength= uiLength;
}

GLTexture1DPtr RayCaster::GetPseudoColorTexture(unsigned int& uiLength) const
{
    uiLength = m_uiPseudoColorLength;
    return m_pPseudoColorTexture;
}

void RayCaster::SetPseudoColorArray(unsigned char* pArray , unsigned int uiLength)
{
    m_pPseudoColorArray = pArray;
    m_uiPseudoColorLength = uiLength;
}

void RayCaster::SetTransferFunctionTexture(GLTexture1DArrayPtr pTexArray)
{
    m_pTransferFunction = pTexArray;
}

void RayCaster::SetSilhouetteEnhancement()
{

}

void RayCaster::SetBoundaryEnhancement()
{

}

void RayCaster::SetMaterial()
{

}

void RayCaster::SetLightColor()
{

}

void RayCaster::SetLightFactor()
{

}

void RayCaster::SetSSDGray(float fGray)
{
    m_fSSDGray = fGray;
}

void RayCaster::SetJitteringEnabled(bool bFlag)
{
    m_bJitteringEnabled = bFlag;
}

void RayCaster::SetBounding(const Vector3f& vMin, const Vector3f& vMax)
{
    m_vBoundingMin = vMin;
    m_vBoundingMax = vMax;
}

void RayCaster::SetClippingPlaneFunction(const std::vector<Vector4f> &vecFunc)
{
    m_vClippingPlaneFunc = vecFunc;
}

void RayCaster::SetMaskMode(MaskMode eMode)
{
    m_eMaskMode = eMode;
}

void RayCaster::SetCompositeMode(CompositeMode eMode)
{
    m_eCompositeMode = eMode;
}

void RayCaster::SetInterpolationMode(InterpolationMode eMode)
{
    m_eInterpolationMode = eMode;
}

void RayCaster::SetShadingMode(ShadingMode eMode)
{
    m_eShadingMode = eMode;
}

void RayCaster::SetColorInverseMode(ColorInverseMode eMode)
{
    m_eColorInverseMode = eMode;
}

void RayCaster::SetCanvas(std::shared_ptr<RayCasterCanvas> pCanvas)
{
    m_pCanvas = pCanvas;
}

void RayCaster::SetStrategy(RayCastingStrategy eS)
{
    m_eStrategy = eS;
}

void RayCaster::SetBrickSize(unsigned int uiBrickSize)
{
    m_uiBrickSize = uiBrickSize;
}

void RayCaster::SetBrickExpand(unsigned int uiBrickExpand)
{
    m_uiBrickExpand = uiBrickExpand;
}

void RayCaster::SetBrickCorner(BrickCorner* pBC)
{
    m_pBrickCorner = pBC;
}

void RayCaster::SetVolumeBrickUnit(BrickUnit* pBU)
{
    m_pVolumeBrickUnit = pBU;
}

void RayCaster::SetMaskBrickUnit(BrickUnit* pBU)
{
    m_pMaskBrickUnit = pBU;
}

void RayCaster::SetMaskBrickInfo(MaskBrickInfo* pMBI)
{
    m_pMaskBrickInfo = pMBI;
}

void RayCaster::SetVolumeBrickInfo(VolumeBrickInfo* pVBI)
{
    m_pVolumeBrickInfo = pVBI;
}

const std::vector<BrickDistance>& RayCaster::GetBrickDistance() const
{
    return m_pRayCastingCPUBrickAcc->GetBrickDistance();
}

unsigned int RayCaster::GetRayCastingBrickCount() const
{
    return m_pRayCastingCPUBrickAcc->GetRayCastingBrickCount();
}

std::shared_ptr<ImageData> RayCaster::GetVolumeData()
{
    return m_pVolumeData;
}

std::vector<GLTexture3DPtr> RayCaster::GetVolumeDataTexture()
{
    return m_vecVolumeDataTex;
}

float RayCaster::GetSampleRate() const
{
    return m_fSampleRate;
}

void RayCaster::GetGlobalWindowLevel(float& fWW , float& fWL) const
{
    fWW = m_fGlobalWW;
    fWL = m_fGlobalWL;
}

std::shared_ptr<EntryExitPoints> RayCaster::GetEntryExitPoints() const
{
    return m_pEntryExitPoints;
}



MED_IMAGING_END_NAMESPACE