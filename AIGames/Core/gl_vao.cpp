#include "gl_vao.h"



GLVAO::GLVAO(UIDType uid):GLObject(uid),_vao_id(0)
{
    set_type("GLVAO");
}

GLVAO::~GLVAO()
{

}

void GLVAO::initialize()
{
    if (0 == _vao_id)
    {
        glGenVertexArrays(1 , &_vao_id);
    }
}

void GLVAO::finalize()
{
    if (0 != _vao_id)
    {
        glDeleteVertexArrays(1 , &_vao_id);
    }
}

unsigned int GLVAO::get_id() const
{
    return _vao_id;
}

void GLVAO::bind()
{
    glBindVertexArray(_vao_id);
}

void GLVAO::unbind()
{
    glBindVertexArray(0);
}

