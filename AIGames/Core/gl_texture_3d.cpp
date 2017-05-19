#include "gl_texture_3d.h"



GLTexture3D::GLTexture3D(UIDType uid) : GLTextureBase(uid),
_width(0),
_height(0),
_depth(0),
_format(GL_RGBA),
_internal_format(GL_RGBA8),
_type(GL_UNSIGNED_BYTE)
{
    set_type("GLTexture3D");
}

GLTexture3D::~GLTexture3D()
{

}

void GLTexture3D::bind()
{
    glBindTexture(GL_TEXTURE_3D , _texture_id);
}

void GLTexture3D::unbind()
{
    glBindTexture(GL_TEXTURE_3D , 0);
}

void GLTexture3D::load(GLint internalformat , GLsizei width, GLsizei height, GLsizei depth,GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage3D(GL_TEXTURE_3D , level , internalformat ,width , height , depth , 0, format ,type ,data);
    _width = width;
    _height = height;
    _depth = depth;
    _format = format;
    _internal_format = internalformat;
    _type= type;
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
    return _width;
}

GLsizei GLTexture3D::get_height()
{
    return _height;
}

GLsizei GLTexture3D::get_depth()
{
    return _depth;
}

GLenum GLTexture3D::get_format()
{
    return _format;
}

