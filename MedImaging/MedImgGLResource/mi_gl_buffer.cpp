#include "mi_gl_buffer.h"

MED_IMAGING_BEGIN_NAMESPACE

GLBuffer::GLBuffer(UIDType uid):GLObject(uid)
,m_eTarget(GL_ARRAY_BUFFER)
,m_uiBufferID(0)
{
    SetType("GLBuffer");
}

GLBuffer::~GLBuffer()
{

}

void GLBuffer::SetBufferTarget(GLenum target)
{
    m_eTarget = target;
}

GLenum GLBuffer::GetBufferTarget() const
{
    return m_eTarget;
}

void GLBuffer::Initialize()
{
    if (0 == m_uiBufferID)
    {
        glGenBuffers(1 , &m_uiBufferID);
    }
}

void GLBuffer::Finalize()
{
    if (0 != m_uiBufferID)
    {  
        glDeleteBuffers(1 , &m_uiBufferID);
        m_uiBufferID = 0;
    }
}

unsigned int GLBuffer::GetID() const
{
    return m_uiBufferID;
}

void GLBuffer::Bind()
{
    if (0 == m_uiBufferID)
    {
        GLRESOURCE_THROW_EXCEPTION("Invalid buffer!");
    }
    glBindBuffer(m_eTarget , m_uiBufferID);
}

void GLBuffer::UnBind()
{
    glBindBuffer(m_eTarget , 0);
}

void GLBuffer::Load(GLsizei size, const void* data, GLenum usage)
{
    glBufferData(m_eTarget , size , data, usage);
}



MED_IMAGING_END_NAMESPACE