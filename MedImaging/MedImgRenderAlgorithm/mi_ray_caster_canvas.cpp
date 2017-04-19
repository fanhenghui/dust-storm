#include "mi_ray_caster_canvas.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCasterCanvas::RayCasterCanvas():m_bInit(false),m_iWidth(32),m_iHeight(32)/*,m_eDataType(USHORT)*/
{

}

RayCasterCanvas::~RayCasterCanvas()
{
    Finialize();
}

void RayCasterCanvas::Initialize()
{
    if (!m_bInit)
    {
        CHECK_GL_ERROR

        UIDType idFBO=0;
        m_pFBO = GLResourceManagerContainer::Instance()->GetFBOManager()->CreateObject(idFBO);
        m_pFBO->Initialize();
        m_pFBO->SetTraget(GL_FRAMEBUFFER);

        UIDType idTexColor = 0;
        m_pColorAttach0 = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(idTexColor);
        m_pColorAttach0->Initialize();
        m_pColorAttach0->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_EDGE);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        m_pColorAttach0->Load(GL_RGBA8 , m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        /*UIDType idTexGray = 0;
        m_pGrayAttach1 = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(idTexGray);
        m_pGrayAttach1->Initialize();
        m_pGrayAttach1->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_EDGE);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        GLenum eInteranlForamt , eFormat , eDataType;
        GLUtils::GetGrayTextureFormat(m_eDataType , eInteranlForamt , eFormat , eDataType);
        m_pGrayAttach1->Load(eInteranlForamt , m_iWidth , m_iHeight , eFormat , eDataType , nullptr);*/

        UIDType idTexDepth = 0;
        m_pDepthAttach = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(idTexDepth);
        m_pDepthAttach->Initialize();
        m_pDepthAttach->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_EDGE);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        m_pDepthAttach->Load(GL_DEPTH_COMPONENT16 , m_iWidth , m_iHeight , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

        //Bind texture to FBO
        m_pFBO->Bind();
        
        m_pFBO->AttachTexture(GL_COLOR_ATTACHMENT0 , m_pColorAttach0);
        //m_pFBO->AttachTexture(GL_COLOR_ATTACHMENT1 , m_pGrayAttach1);
        m_pFBO->AttachTexture(GL_DEPTH_ATTACHMENT , m_pDepthAttach);

        m_pFBO->UnBind();

        CHECK_GL_ERROR;

        //Create array
        m_pColorArray.reset(new RGBAUnit[m_iWidth*m_iHeight]);

        /*if (UCHAR == m_eDataType || CHAR == m_eDataType)
        {
        m_pGrayArray.reset(new char[m_iWidth*m_iHeight]);
        }
        else if (USHORT == m_eDataType || SHORT == m_eDataType)
        {
        m_pGrayArray.reset(new char[m_iWidth*m_iHeight*sizeof(short)]);
        }
        else if (FLOAT == m_eDataType)
        {
        m_pGrayArray.reset(new char[m_iWidth*m_iHeight*sizeof(float)]);
        }
        else
        {
        RENDERALGO_THROW_EXCEPTION("Invalid data type!");
        }*/

        m_bInit = true;
    }
}

void RayCasterCanvas::Finialize()
{
    if (m_bInit)
    {
        GLResourceManagerContainer::Instance()->GetFBOManager()->RemoveObject(m_pFBO->GetUID());
        GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemoveObject(m_pColorAttach0->GetUID());
        //GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemovOebject(m_pGrayAttach1->GetUID());
        GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemoveObject(m_pDepthAttach->GetUID());

        GLResourceManagerContainer::Instance()->GetFBOManager()->Update();
        GLResourceManagerContainer::Instance()->GetTexture2DManager()->Update();
        m_bInit = false;
    }
}

void RayCasterCanvas::SetDisplaySize( int iWidth , int iHeight )
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;
}

GLFBOPtr RayCasterCanvas::GetFBO()
{
    return m_pFBO;
}

RGBAUnit* RayCasterCanvas::GetColorArray()
{
    return m_pColorArray.get();
}

//void* RayCasterCanvas::GetGrayArray()
//{
//    return m_pGrayArray.get();
//}

void RayCasterCanvas::UpdateFBO()
{
    if (m_bInit)
    {
        m_pColorAttach0->Bind();
        m_pColorAttach0->Load(GL_RGBA8 , m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        /*m_pGrayAttach1->Bind();
        GLenum eInteranlForamt , eFormat , eDataType;
        GLUtils::GetGrayTextureFormat(m_eDataType , eInteranlForamt , eFormat , eDataType);
        m_pGrayAttach1->Load(eInteranlForamt , m_iWidth , m_iHeight , eFormat , eDataType , nullptr);*/

        m_pDepthAttach->Bind();
        m_pDepthAttach->Load(GL_DEPTH_COMPONENT16 , m_iWidth , m_iHeight , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

        /*m_pColorArray.reset(new RGBAUnit[m_iWidth*m_iHeight]);
        if (UCHAR == m_eDataType || CHAR == m_eDataType)
        {
        m_pGrayArray.reset(new char[m_iWidth*m_iHeight]);
        }
        else if (USHORT == m_eDataType || SHORT == m_eDataType)
        {
        m_pGrayArray.reset(new char[m_iWidth*m_iHeight*sizeof(short)]);
        }
        else if (FLOAT == m_eDataType)
        {
        m_pGrayArray.reset(new char[m_iWidth*m_iHeight*sizeof(float)]);
        }
        else
        {
        RENDERALGO_THROW_EXCEPTION("Invalid data type!");
        }*/
    }
}

//void RayCasterCanvas::SetDataType(DataType eDataType)
//{
//    m_eDataType = eDataType;
//}

void RayCasterCanvas::UploadColorArray()
{
    CHECK_GL_ERROR
    m_pColorAttach0->Bind();
    m_pColorAttach0->Update(0,0,m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , m_pColorArray.get());
    CHECK_GL_ERROR
}

//void RayCasterCanvas::DownloadGrayArray()
//{
//    GLenum eInteranlForamt , eFormat , eDataType;
//    GLUtils::GetGrayTextureFormat(m_eDataType , eInteranlForamt , eFormat , eDataType);
//    m_pGrayAttach1->Bind();
//    m_pGrayAttach1->Download(eFormat , eDataType , m_pGrayArray.get());
//}

GLTexture2DPtr RayCasterCanvas::GetColorAttachTexture()
{
    return m_pColorAttach0;
}

//GLTexture2DPtr RayCasterCanvas::GetGrayAttachTexture()
//{
//    return m_pGrayAttach1;
//}

void RayCasterCanvas::DebugOutputColor(const std::string& sFileName)
{
    m_pColorAttach0->Bind();
    std::unique_ptr<unsigned char[]> pRGBA(new unsigned char[m_iWidth*m_iHeight*4]);
    m_pColorAttach0->Download(GL_RGBA , GL_UNSIGNED_BYTE , pRGBA.get());

    std::ofstream out(sFileName , std::ios::out | std::ios::binary);
    if (out.is_open())
    {
        out.write((char*)pRGBA.get(), m_iWidth*m_iHeight*4);
    }
    out.close();

    /*std::ofstream out(sFileName , std::ios::out | std::ios::binary);
    if (out.is_open())
    {
        out.write((char*)m_pColorArray.get(), m_iWidth*m_iHeight*4);
    }
    out.close();*/
}

//void RayCasterCanvas::DebugOutputGray(const std::string& sFileName)
//{
//
//}

void RayCasterCanvas::GetDisplaySize(int& iWidth, int& iHeight) const
{
    iWidth = m_iWidth;
    iHeight = m_iHeight;
}

MED_IMAGING_END_NAMESPACE