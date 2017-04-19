#include "mi_gl_texture_base.h"

MED_IMAGING_BEGIN_NAMESPACE

GLTextureBase::GLTextureBase(UIDType uid) : GLObject(uid) , m_uiTextueID(0)
{
    SetType("GLTextureBase");
}

GLTextureBase::~GLTextureBase()
{

}

void GLTextureBase::Initialize()
{
    if (0 == m_uiTextueID)
    {
        glGenTextures(1 , &m_uiTextueID);
    }
}

void GLTextureBase::Finalize()
{
    if (0 != m_uiTextueID)
    {
        glDeleteTextures(1 , &m_uiTextueID);
        m_uiTextueID = 0;
    }
}

unsigned int GLTextureBase::GetID() const
{
    return m_uiTextueID;
}

void GLTextureBase::BindImage(GLuint unit , GLint level , GLboolean layered , GLint layer , GLenum access, GLenum format)
{
    glBindImageTexture(unit , m_uiTextueID , level , layered , layer , access , format);
}

MED_IMAGING_END_NAMESPACE