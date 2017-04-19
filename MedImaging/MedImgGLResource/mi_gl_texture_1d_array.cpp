#include "mi_gl_texture_1d_array.h"

MED_IMAGING_BEGIN_NAMESPACE

GLTexture1DArray::GLTexture1DArray(UIDType uid): GLTextureBase(uid),
m_iWidth(0),
m_iArraySize(0),
m_eFormat(GL_RGBA),
m_eInternalFormat(GL_RGBA8),
m_eType(GL_UNSIGNED_BYTE)
{
    SetType("GLTexture1DArray");
}

GLTexture1DArray::~GLTexture1DArray()
{

}

void GLTexture1DArray::Bind()
{
    glBindTexture(GL_TEXTURE_1D_ARRAY , m_uiTextueID);
}

void GLTexture1DArray::UnBind()
{
    glBindTexture(GL_TEXTURE_1D_ARRAY , 0);
}

void GLTexture1DArray::Load(GLint internalformat , GLsizei width, GLsizei arraysize , GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage2D(GL_TEXTURE_1D_ARRAY , level , internalformat , width ,arraysize , 0 , format , type , data);
    m_iWidth = width;
    m_iArraySize = arraysize;
    m_eFormat = format;
    m_eInternalFormat = internalformat;
    m_eType= type;
}

void GLTexture1DArray::Update(GLint xoffset , GLsizei width , GLsizei arrayidx, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexSubImage2D(GL_TEXTURE_1D_ARRAY , level , xoffset , arrayidx , width ,1 , format ,type ,data);
}

void GLTexture1DArray::Download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_1D_ARRAY , level ,format ,type ,buffer );
}

int GLTexture1DArray::GetWidth()
{
    return m_iWidth;
}

int GLTexture1DArray::GetArraySize()
{
    return m_iArraySize;
}

GLenum GLTexture1DArray::GetFormat()
{
    return m_eFormat;
}

MED_IMAGING_END_NAMESPACE
