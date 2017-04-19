#include "mi_scene_base.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

SceneBase::SceneBase():m_iWidth(128),m_iHeight(128),m_bDirty(true),m_sName("Scene")
{
}

SceneBase::SceneBase(int iWidth , int iHeight):m_iWidth(iWidth) , m_iHeight(iHeight),m_bDirty(true)
{
}

SceneBase::~SceneBase()
{
    Finalize();
}

void SceneBase::RenderToBack()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pSceneFBO->GetID());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glDrawBuffer(GL_BACK);
    glBlitFramebuffer(0,0,m_iWidth, m_iHeight , 0,0,m_iWidth , m_iHeight , GL_COLOR_BUFFER_BIT , GL_NEAREST);
}

std::shared_ptr<CameraBase> SceneBase::GetCamera()
{
    return m_pCamera;
}

void SceneBase::Initialize()
{
    if (!m_pSceneFBO)
    {
        //Init FBO
        CHECK_GL_ERROR;

        UIDType idFBO=0;
        m_pSceneFBO = GLResourceManagerContainer::Instance()->GetFBOManager()->CreateObject(idFBO);
        m_pSceneFBO->SetDescription("Scene base FBO");
        m_pSceneFBO->Initialize();
        m_pSceneFBO->SetTraget(GL_FRAMEBUFFER);

        UIDType idTexColor = 0;
        m_pSceneColorAttach0 = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(idTexColor);
        m_pSceneColorAttach0->SetDescription("Scene base Color Attachment 0");
        m_pSceneColorAttach0->Initialize();
        m_pSceneColorAttach0->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_EDGE);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        m_pSceneColorAttach0->Load(GL_RGBA8 , m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        UIDType idTexDepth = 0;
        m_pSceneDepthAttach = GLResourceManagerContainer::Instance()->GetTexture2DManager()->CreateObject(idTexDepth);
        m_pSceneDepthAttach->SetDescription("Scene base Depth Attachment");
        m_pSceneDepthAttach->Initialize();
        m_pSceneDepthAttach->Bind();
        GLTextureUtils::Set2DWrapST(GL_CLAMP_TO_EDGE);
        GLTextureUtils::SetFilter(GL_TEXTURE_2D , GL_LINEAR);
        m_pSceneDepthAttach->Load(GL_DEPTH_COMPONENT16 , m_iWidth , m_iHeight , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

        //Bind texture to FBO
        m_pSceneFBO->Bind();

        m_pSceneFBO->AttachTexture(GL_COLOR_ATTACHMENT0 , m_pSceneColorAttach0);
        m_pSceneFBO->AttachTexture(GL_DEPTH_ATTACHMENT , m_pSceneDepthAttach);

        m_pSceneFBO->UnBind();

        CHECK_GL_ERROR;
    }
}

void SceneBase::Finalize()
{
    GLResourceManagerContainer::Instance()->GetFBOManager()->RemoveObject(m_pSceneFBO->GetUID());
    GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemoveObject(m_pSceneColorAttach0->GetUID());
    GLResourceManagerContainer::Instance()->GetTexture2DManager()->RemoveObject(m_pSceneDepthAttach->GetUID());

    GLResourceManagerContainer::Instance()->GetFBOManager()->Update();
    GLResourceManagerContainer::Instance()->GetTexture2DManager()->Update();
}

void SceneBase::SetDisplaySize(int iWidth , int iHeight)
{
    m_iWidth = iWidth;
    m_iHeight = iHeight;

    m_pSceneColorAttach0->Bind();
    m_pSceneColorAttach0->Load(GL_RGBA8 , m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

    m_pSceneDepthAttach->Bind();
    m_pSceneDepthAttach->Load(GL_DEPTH_COMPONENT16 , m_iWidth , m_iHeight , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

    SetDirty(true);
}

void SceneBase::Render(int)
{

}

void SceneBase::Rotate(const Point2& ptPre , const Point2& ptCur)
{

}

void SceneBase::Zoom(const Point2& ptPre , const Point2& ptCur)
{

}

void SceneBase::Pan(const Point2& ptPre , const Point2& ptCur)
{

}

void SceneBase::GetDisplaySize(int& iWidth, int& iHeight) const
{
    iWidth = m_iWidth;
    iHeight = m_iHeight;
}

void SceneBase::SetDirty(bool bFlag)
{
    m_bDirty = bFlag;
}

bool SceneBase::GetDirty() const
{
    return m_bDirty;
}

void SceneBase::SetName(const std::string& sName)
{
    m_sName = sName;
}

const std::string& SceneBase::GetName() const
{
    return m_sName;
}


MED_IMAGING_END_NAMESPACE