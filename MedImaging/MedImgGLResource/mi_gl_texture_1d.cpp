#include "mi_gl_texture_1d.h"

MED_IMAGING_BEGIN_NAMESPACE

GLTexture1D::GLTexture1D(UIDType uid) : GLTextureBase(uid),
m_iWidth(0),
m_eFormat(GL_RGBA),
m_eInternalFormat(GL_RGBA8),
m_eType(GL_UNSIGNED_BYTE)
{
    SetType("GLTexture1D");
}

GLTexture1D::~GLTexture1D()
{

}

void GLTexture1D::Bind()
{
    glBindTexture(GL_TEXTURE_1D , m_uiTextueID);
}

void GLTexture1D::UnBind()
{
    glBindTexture(GL_TEXTURE_1D , 0);
}

void GLTexture1D::Load(GLint internalformat , GLsizei width, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage1D(GL_TEXTURE_1D , level , internalformat ,width , 0, format ,type ,data);
    m_iWidth = width;
    m_eFormat = format;
    m_eInternalFormat = internalformat;
    m_eType= type;
}

void GLTexture1D::Update(GLint xoffset , GLsizei width , GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
     glTexSubImage1D(GL_TEXTURE_1D , level , xoffset ,width ,format ,type ,data);
}

void GLTexture1D::Download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_1D , level ,format ,type ,buffer );
}

GLsizei GLTexture1D::GetWidth()
{
    return m_iWidth;
}

GLenum GLTexture1D::GetFormat()
{
    return m_eFormat;
}

MED_IMAGING_END_NAMESPACE