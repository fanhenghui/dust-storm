#include "mi_gl_vao.h"

MED_IMAGING_BEGIN_NAMESPACE

GLVAO::GLVAO(UIDType uid):GLObject(uid),m_uiVAOID(0)
{
    SetType("GLVAO");
}

GLVAO::~GLVAO()
{

}

void GLVAO::Initialize()
{
    if (0 == m_uiVAOID)
    {
        glGenVertexArrays(1 , &m_uiVAOID);
    }
}

void GLVAO::Finalize()
{
    if (0 != m_uiVAOID)
    {
        glDeleteVertexArrays(1 , &m_uiVAOID);
    }
}

unsigned int GLVAO::GetID() const
{
    return m_uiVAOID;
}

void GLVAO::Bind()
{
    glBindVertexArray(m_uiVAOID);
}

void GLVAO::UnBind()
{
    glBindVertexArray(0);
}

MED_IMAGING_END_NAMESPACE