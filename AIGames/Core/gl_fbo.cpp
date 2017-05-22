#include "gl_fbo.h"
#include "Core/gl_texture_2d.h"
#include "Core/gl_utils.h"



GLFBO::GLFBO(UIDType uid):GLObject(uid),_fbo_id(0),_target(GL_FRAMEBUFFER)
{
    set_type("GLFBO");
}

GLFBO::~GLFBO()
{

}

void GLFBO::initialize()
{
    if (0 == _fbo_id)
    {
        glGenFramebuffers(1 , &_fbo_id);
    }
}

void GLFBO::finalize()
{
    if (0 != _fbo_id)
    {
        glDeleteFramebuffers(1 , &_fbo_id);
    }
}

unsigned int GLFBO::get_id() const
{
    return _fbo_id;
}

void GLFBO::bind()
{
    glBindFramebuffer(_target , _fbo_id);
}

void GLFBO::unbind()
{
    glBindFramebuffer(_target , 0);
}

void GLFBO::set_target(GLenum target)
{
    _target = target;
}

GLenum GLFBO::get_target()
{
    return _target;
}

void GLFBO::attach_texture( GLenum attachment , std::shared_ptr<GLTexture2D> tex )
{
    glFramebufferTexture2D(_target , attachment , GL_TEXTURE_2D , tex->get_id() , 0);
}

