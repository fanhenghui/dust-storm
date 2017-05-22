#include "gl_texture_1d.h"



GLTexture1D::GLTexture1D(UIDType uid) : GLTextureBase(uid),
_width(0),
_format(GL_RGBA),
_internal_format(GL_RGBA8),
_type(GL_UNSIGNED_BYTE)
{
    set_type("GLTexture1D");
}

GLTexture1D::~GLTexture1D()
{

}

void GLTexture1D::bind()
{
    glBindTexture(GL_TEXTURE_1D , _texture_id);
}

void GLTexture1D::unbind()
{
    glBindTexture(GL_TEXTURE_1D , 0);
}

void GLTexture1D::load(GLint internalformat , GLsizei width, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage1D(GL_TEXTURE_1D , level , internalformat ,width , 0, format ,type ,data);
    _width = width;
    _format = format;
    _internal_format = internalformat;
    _type= type;
}

void GLTexture1D::update(GLint xoffset , GLsizei width , GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
     glTexSubImage1D(GL_TEXTURE_1D , level , xoffset ,width ,format ,type ,data);
}

void GLTexture1D::download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_1D , level ,format ,type ,buffer );
}

GLsizei GLTexture1D::get_width()
{
    return _width;
}

GLenum GLTexture1D::get_format()
{
    return _format;
}

