#include "gl_texture_base.h"



GLTextureBase::GLTextureBase(UIDType uid) : GLObject(uid) , _texture_id(0)
{
    set_type("GLTextureBase");
}

GLTextureBase::~GLTextureBase()
{

}

void GLTextureBase::initialize()
{
    if (0 == _texture_id)
    {
        glGenTextures(1 , &_texture_id);
    }
}

void GLTextureBase::finalize()
{
    if (0 != _texture_id)
    {
        glDeleteTextures(1 , &_texture_id);
        _texture_id = 0;
    }
}

unsigned int GLTextureBase::get_id() const
{
    return _texture_id;
}

void GLTextureBase::bind_image(GLuint unit , GLint level , GLboolean layered , GLint layer , GLenum access, GLenum format)
{
    glBindImageTexture(unit , _texture_id , level , layered , layer , access , format);
}

