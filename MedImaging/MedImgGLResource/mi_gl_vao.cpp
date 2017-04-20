#include "mi_gl_vao.h"

MED_IMAGING_BEGIN_NAMESPACE

GLVAO::GLVAO(UIDType uid):GLObject(uid),m_uiVAOID(0)
{
    set_type("GLVAO");
}

GLVAO::~GLVAO()
{

}

void GLVAO::initialize()
{
    if (0 == m_uiVAOID)
    {
        glGenVertexArrays(1 , &m_uiVAOID);
    }
}

void GLVAO::finalize()
{
    if (0 != m_uiVAOID)
    {
        glDeleteVertexArrays(1 , &m_uiVAOID);
    }
}

unsigned int GLVAO::get_id() const
{
    return m_uiVAOID;
}

void GLVAO::bind()
{
    glBindVertexArray(m_uiVAOID);
}

void GLVAO::unbind()
{
    glBindVertexArray(0);
}

MED_IMAGING_END_NAMESPACE