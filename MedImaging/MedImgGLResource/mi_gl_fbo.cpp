#include "mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

GLFBO::GLFBO(UIDType uid):GLObject(uid),m_uiFBOID(0),m_eTarget(GL_FRAMEBUFFER)
{
    SetType("GLFBO");
}

GLFBO::~GLFBO()
{

}

void GLFBO::Initialize()
{
    if (0 == m_uiFBOID)
    {
        glGenFramebuffers(1 , &m_uiFBOID);
    }
}

void GLFBO::Finalize()
{
    if (0 != m_uiFBOID)
    {
        glDeleteFramebuffers(1 , &m_uiFBOID);
    }
}

unsigned int GLFBO::GetID() const
{
    return m_uiFBOID;
}

void GLFBO::Bind()
{
    glBindFramebuffer(m_eTarget , m_uiFBOID);
}

void GLFBO::UnBind()
{
    glBindFramebuffer(m_eTarget , 0);
}

void GLFBO::SetTraget(GLenum eTarget)
{
    m_eTarget = eTarget;
}

GLenum GLFBO::GetTraget()
{
    return m_eTarget;
}

void GLFBO::AttachTexture( GLenum eAttach , std::shared_ptr<GLTexture2D> pTex2D )
{
    glFramebufferTexture2D(m_eTarget , eAttach , GL_TEXTURE_2D , pTex2D->GetID() , 0);
}

MED_IMAGING_END_NAMESPACE