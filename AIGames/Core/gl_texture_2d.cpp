#include "gl_texture_2d.h"



GLTexture2D::GLTexture2D(UIDType uid) : GLTextureBase(uid),
_width(0),
_height(0),
_format(GL_RGBA),
_internal_format(GL_RGBA8),
_type(GL_UNSIGNED_BYTE)
{
    set_type("GLTexture2D");
}

GLTexture2D::~GLTexture2D()
{

}

void GLTexture2D::bind()
{
    glBindTexture(GL_TEXTURE_2D , _texture_id);
}

void GLTexture2D::unbind()
{
    glBindTexture(GL_TEXTURE_2D , 0);
}

void GLTexture2D::load(GLint internalformat , GLsizei width, GLsizei height, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage2D(GL_TEXTURE_2D , level , internalformat ,width , height , 0, format ,type ,data);
    _width = width;
    _height = height;
    _format = format;
    _internal_format = internalformat;
    _type= type;
}

void GLTexture2D::update(GLint xoffset , GLint yoffset ,GLsizei width , GLsizei height, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexSubImage2D(GL_TEXTURE_2D , level , xoffset ,yoffset , width ,height , format ,type ,data);
}

void GLTexture2D::download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_2D , level ,format ,type ,buffer );
}

GLsizei GLTexture2D::get_width()
{
    return _width;
}

GLsizei GLTexture2D::get_height()
{
    return _height;
}

GLenum GLTexture2D::get_format()
{
    return _format;
}

