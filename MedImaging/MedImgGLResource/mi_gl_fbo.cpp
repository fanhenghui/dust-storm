#include "mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

GLFBO::GLFBO(UIDType uid):GLObject(uid),m_uiFBOID(0),m_eTarget(GL_FRAMEBUFFER)
{
    set_type("GLFBO");
}

GLFBO::~GLFBO()
{

}

void GLFBO::initialize()
{
    if (0 == m_uiFBOID)
    {
        glGenFramebuffers(1 , &m_uiFBOID);
    }
}

void GLFBO::finalize()
{
    if (0 != m_uiFBOID)
    {
        glDeleteFramebuffers(1 , &m_uiFBOID);
    }
}

unsigned int GLFBO::get_id() const
{
    return m_uiFBOID;
}

void GLFBO::bind()
{
    glBindFramebuffer(m_eTarget , m_uiFBOID);
}

void GLFBO::unbind()
{
    glBindFramebuffer(m_eTarget , 0);
}

void GLFBO::set_target(GLenum eTarget)
{
    m_eTarget = eTarget;
}

GLenum GLFBO::get_target()
{
    return m_eTarget;
}

void GLFBO::attach_texture( GLenum eAttach , std::shared_ptr<GLTexture2D> pTex2D )
{
    glFramebufferTexture2D(m_eTarget , eAttach , GL_TEXTURE_2D , pTex2D->get_id() , 0);
}

MED_IMAGING_END_NAMESPACE