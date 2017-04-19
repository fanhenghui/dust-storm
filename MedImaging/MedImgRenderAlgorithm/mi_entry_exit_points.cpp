#include "mi_entry_exit_points.h"

#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgArithmetic/mi_camera_base.h"

#include "mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

EntryExitPoints::EntryExitPoints():m_iWidth(4),m_iHeight(4),m_bInit(false),m_eStrategy(CPU_BASE)
{
    m_pEntryBuffer.reset(new Vector4f[m_iWidth*m_iHeight]);
    m_pExitBuffer.reset(new Vector4f[m_iWidth*m_iHeight]);
    UIDType uid;
    m_pEntryTex = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(uid);
    m_pExitTex = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(uid);
}

void EntryExitPoints::Initialize()
{
    if (!m_bInit)
    {
        m_pEntryTex->Initialize();
        m_pExitTex->Initialize();
        m_bInit = true;
    }
}

void EntryExitPoints::Finialize()
{
    if (m_bInit)
    {
        GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemoveObject(m_pEntryTex->GetUID());
        GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemoveObject(m_pExitTex->GetUID());
        GLResourceManagerContainer::Instance()->GetTexture2DManager()->Update();
        m_bInit = false;
    }
}

EntryExitPoints::~EntryExitPoints()
{

}

void EntryExitPoints::SetDisplaySize(int iWidth , int iHeight)
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;
    m_pEntryBuffer.reset(new Vector4f[m_iWidth*m_iHeight]);
    m_pExitBuffer.reset(new Vector4f[m_iWidth*m_iHeight]);

    //Resize texture
    if (GPU_BASE == m_eStrategy)
    { 
        Initialize();

        CHECK_GL_ERROR;

        m_pEntryTex->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_BORDER);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        m_pEntryTex->Load(GL_RGBA32F , m_iWidth , m_iHeight , GL_RGBA , GL_FLOAT , NULL);
        m_pEntryTex->UnBind();

        m_pExitTex->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_BORDER);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        m_pExitTex->Load(GL_RGBA32F , m_iWidth , m_iHeight , GL_RGBA , GL_FLOAT , NULL);
        m_pExitTex->UnBind();

        CHECK_GL_ERROR;
    }
}

void EntryExitPoints::GetDisplaySize(int& iWidth , int& iHeight)
{
    iWidth = m_iWidth;
    iHeight = m_iHeight;
}

std::shared_ptr<GLTexture2D> EntryExitPoints::GetEntryPointsTexture()
{
    return m_pEntryTex;
}

std::shared_ptr<GLTexture2D> EntryExitPoints::GetExitPointsTexture()
{
    return m_pExitTex;
}

Vector4f* EntryExitPoints::GetEntryPointsArray()
{
    return m_pEntryBuffer.get();
}

Vector4f* EntryExitPoints::GetExitPointsArray()
{
    return m_pExitBuffer.get();
}

void EntryExitPoints::SetImageData(std::shared_ptr<ImageData> pImgData)
{
    m_pImgData = pImgData;
}

void EntryExitPoints::SetCamera(std::shared_ptr<CameraBase> pCamera)
{
    m_pCamera = pCamera;
}

void EntryExitPoints::SetCameraCalculator(std::shared_ptr<CameraCalculator> pCameraCal)
{
    m_pCameraCalculator = pCameraCal;
}

void EntryExitPoints::DebugOutputEntryPoints(const std::string& sFileName)
{
    Vector4f* pPoints = m_pEntryBuffer.get();
    std::ofstream out(sFileName , std::ios::binary | std::ios::out);
    if (out.is_open())
    {
        std::unique_ptr<unsigned char[]> pRGB(new unsigned char[m_iWidth*m_iHeight*3]);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pImgData);
        unsigned int *uiDim = m_pImgData->m_uiDim;
        float fDimR[3] = { 1.0f/(float)uiDim[0],1.0f/(float)uiDim[1],1.0f/(float)uiDim[2]};
        unsigned char r,g,b;
        float fR , fG , fB;
        for (int i = 0 ; i < m_iWidth*m_iHeight ; ++i)
        {
            fR =pPoints[i]._m[0] *fDimR[0]*255.0f;
            fG =pPoints[i]._m[1] *fDimR[1]*255.0f;
            fB =pPoints[i]._m[2] *fDimR[2]*255.0f;

            fR = fR > 255.0f ? 255.0f : fR;
            fR = fR <0.0f ? 0.0f : fR;

            fG = fG > 255.0f ? 255.0f : fG;
            fG = fG <0.0f ? 0.0f : fG;

            fB = fB > 255.0f ? 255.0f : fB;
            fB = fB <0.0f ? 0.0f : fB;

            r = unsigned char(fR);
            g = unsigned char(fG);
            b = unsigned char(fB);

            pRGB[i*3] = r;
            pRGB[i*3+1] = g;
            pRGB[i*3+2] = b;

        }

        out.write((char*)pRGB.get()  , m_iWidth*m_iHeight*3);
        out.close();
    }
    else
    {
        //TODO LOG
        std::cout << "Open file " << sFileName << " failed!\n";
    }
}

void EntryExitPoints::DebugOutputExitPoints(const std::string& sFileName)
{
    Vector4f* pPoints = m_pExitBuffer.get();
    std::ofstream out(sFileName , std::ios::binary | std::ios::out);
    if (out.is_open())
    {
        std::unique_ptr<unsigned char[]> pRGB(new unsigned char[m_iWidth*m_iHeight*3]);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pImgData);
        unsigned int *uiDim = m_pImgData->m_uiDim;
        float fDimR[3] = { 1.0f/(float)uiDim[0],1.0f/(float)uiDim[1],1.0f/(float)uiDim[2]};
        unsigned char r,g,b;
        float fR , fG , fB;
        for (int i = 0 ; i < m_iWidth*m_iHeight ; ++i)
        {
            fR =pPoints[i]._m[0] *fDimR[0]*255.0f;
            fG =pPoints[i]._m[1] *fDimR[1]*255.0f;
            fB =pPoints[i]._m[2] *fDimR[2]*255.0f;

            fR = fR > 255.0f ? 255.0f : fR;
            fR = fR <0.0f ? 0.0f : fR;

            fG = fG > 255.0f ? 255.0f : fG;
            fG = fG <0.0f ? 0.0f : fG;

            fB = fB > 255.0f ? 255.0f : fB;
            fB = fB <0.0f ? 0.0f : fB;

            r = unsigned char(fR);
            g = unsigned char(fG);
            b = unsigned char(fB);

            pRGB[i*3] = r;
            pRGB[i*3+1] = g;
            pRGB[i*3+2] = b;

        }

        out.write((char*)pRGB.get()  , m_iWidth*m_iHeight*3);
        out.close();
    }
    else
    {
        //TODO LOG
        std::cout << "Open file " << sFileName << " failed!\n";
    }
}

void EntryExitPoints::SetStrategy( RayCastingStrategy eStrategy )
{
    m_eStrategy = eStrategy;
}



MED_IMAGING_END_NAMESPACE