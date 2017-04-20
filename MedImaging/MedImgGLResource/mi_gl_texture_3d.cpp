#include "mi_gl_texture_3d.h"

MED_IMAGING_BEGIN_NAMESPACE

GLTexture3D::GLTexture3D(UIDType uid) : GLTextureBase(uid),
m_iWidth(0),
m_iHeight(0),
m_iDepth(0),
m_eFormat(GL_RGBA),
m_eInternalFormat(GL_RGBA8),
m_eType(GL_UNSIGNED_BYTE)
{
    set_type("GLTexture3D");
}

GLTexture3D::~GLTexture3D()
{

}

void GLTexture3D::bind()
{
    glBindTexture(GL_TEXTURE_3D , m_uiTextueID);
}

void GLTexture3D::unbind()
{
    glBindTexture(GL_TEXTURE_3D , 0);
}

void GLTexture3D::load(GLint internalformat , GLsizei width, GLsizei height, GLsizei depth,GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage3D(GL_TEXTURE_3D , level , internalformat ,width , height , depth , 0, format ,type ,data);
    m_iWidth = width;
    m_iHeight = height;
    m_iDepth = depth;
    m_eFormat = format;
    m_eInternalFormat = internalformat;
    m_eType= type;
}

void GLTexture3D::update(GLint xoffset , GLint yoffset ,GLint zoffset ,GLsizei width , GLsizei height, GLsizei depth,GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexSubImage3D(GL_TEXTURE_3D , level , xoffset ,yoffset ,zoffset , width ,height ,depth, format ,type ,data);
}

void GLTexture3D::download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_3D , level ,format ,type ,buffer );
}

GLsizei GLTexture3D::get_width()
{
    return m_iWidth;
}

GLsizei GLTexture3D::get_height()
{
    return m_iHeight;
}

GLsizei GLTexture3D::get_depth()
{
    return m_iDepth;
}

GLenum GLTexture3D::get_format()
{
    return m_eFormat;
}

MED_IMAGING_END_NAMESPACE