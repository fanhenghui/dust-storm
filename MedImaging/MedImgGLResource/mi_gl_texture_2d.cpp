#include "mi_gl_texture_2d.h"

MED_IMAGING_BEGIN_NAMESPACE

GLTexture2D::GLTexture2D(UIDType uid) : GLTextureBase(uid),
m_iWidth(0),
m_iHeight(0),
m_eFormat(GL_RGBA),
m_eInternalFormat(GL_RGBA8),
m_eType(GL_UNSIGNED_BYTE)
{
    SetType("GLTexture2D");
}

GLTexture2D::~GLTexture2D()
{

}

void GLTexture2D::Bind()
{
    glBindTexture(GL_TEXTURE_2D , m_uiTextueID);
}

void GLTexture2D::UnBind()
{
    glBindTexture(GL_TEXTURE_2D , 0);
}

void GLTexture2D::Load(GLint internalformat , GLsizei width, GLsizei height, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage2D(GL_TEXTURE_2D , level , internalformat ,width , height , 0, format ,type ,data);
    m_iWidth = width;
    m_iHeight = height;
    m_eFormat = format;
    m_eInternalFormat = internalformat;
    m_eType= type;
}

void GLTexture2D::Update(GLint xoffset , GLint yoffset ,GLsizei width , GLsizei height, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexSubImage2D(GL_TEXTURE_2D , level , xoffset ,yoffset , width ,height , format ,type ,data);
}

void GLTexture2D::Download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_2D , level ,format ,type ,buffer );
}

GLsizei GLTexture2D::GetWidth()
{
    return m_iWidth;
}

GLsizei GLTexture2D::GetHeight()
{
    return m_iHeight;
}

GLenum GLTexture2D::GetFormat()
{
    return m_eFormat;
}

MED_IMAGING_END_NAMESPACE