#include "gl_texture_1d_array.h"



GLTexture1DArray::GLTexture1DArray(UIDType uid): GLTextureBase(uid),
_width(0),
_array_size(0),
_format(GL_RGBA),
_internal_format(GL_RGBA8),
_type(GL_UNSIGNED_BYTE)
{
    set_type("GLTexture1DArray");
}

GLTexture1DArray::~GLTexture1DArray()
{

}

void GLTexture1DArray::bind()
{
    glBindTexture(GL_TEXTURE_1D_ARRAY , _texture_id);
}

void GLTexture1DArray::unbind()
{
    glBindTexture(GL_TEXTURE_1D_ARRAY , 0);
}

void GLTexture1DArray::load(GLint internalformat , GLsizei width, GLsizei arraysize , GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexImage2D(GL_TEXTURE_1D_ARRAY , level , internalformat , width ,arraysize , 0 , format , type , data);
    _width = width;
    _array_size = arraysize;
    _format = format;
    _internal_format = internalformat;
    _type= type;
}

void GLTexture1DArray::update(GLint xoffset , GLsizei width , GLsizei arrayidx, GLenum format , GLenum type , const void* data , GLint level /*= 0*/)
{
    glTexSubImage2D(GL_TEXTURE_1D_ARRAY , level , xoffset , arrayidx , width ,1 , format ,type ,data);
}

void GLTexture1DArray::download(GLenum format , GLenum type , void* buffer , GLint level /*= 0*/) const
{
    glGetTexImage(GL_TEXTURE_1D_ARRAY , level ,format ,type ,buffer );
}

int GLTexture1DArray::get_width()
{
    return _width;
}

int GLTexture1DArray::get_array_size()
{
    return _array_size;
}

GLenum GLTexture1DArray::get_format()
{
    return _format;
}


