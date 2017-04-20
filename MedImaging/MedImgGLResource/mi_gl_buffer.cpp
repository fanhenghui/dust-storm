#include "mi_gl_buffer.h"

MED_IMAGING_BEGIN_NAMESPACE

GLBuffer::GLBuffer(UIDType uid):GLObject(uid)
,m_eTarget(GL_ARRAY_BUFFER)
,m_uiBufferID(0)
{
    set_type("GLBuffer");
}

GLBuffer::~GLBuffer()
{

}

void GLBuffer::set_buffer_target(GLenum target)
{
    m_eTarget = target;
}

GLenum GLBuffer::get_buffer_target() const
{
    return m_eTarget;
}

void GLBuffer::initialize()
{
    if (0 == m_uiBufferID)
    {
        glGenBuffers(1 , &m_uiBufferID);
    }
}

void GLBuffer::finalize()
{
    if (0 != m_uiBufferID)
    {  
        glDeleteBuffers(1 , &m_uiBufferID);
        m_uiBufferID = 0;
    }
}

unsigned int GLBuffer::get_id() const
{
    return m_uiBufferID;
}

void GLBuffer::bind()
{
    if (0 == m_uiBufferID)
    {
        GLRESOURCE_THROW_EXCEPTION("Invalid buffer!");
    }
    glBindBuffer(m_eTarget , m_uiBufferID);
}

void GLBuffer::unbind()
{
    glBindBuffer(m_eTarget , 0);
}

void GLBuffer::load(GLsizei size, const void* data, GLenum usage)
{
    glBufferData(m_eTarget , size , data, usage);
}



MED_IMAGING_END_NAMESPACE