#include "gl_buffer.h"



GLBuffer::GLBuffer(UIDType uid):GLObject(uid)
,_target(GL_ARRAY_BUFFER)
,_buffer_id(0)
{
    set_type("GLBuffer");
}

GLBuffer::~GLBuffer()
{

}

void GLBuffer::set_buffer_target(GLenum target)
{
    _target = target;
}

GLenum GLBuffer::get_buffer_target() const
{
    return _target;
}

void GLBuffer::initialize()
{
    if (0 == _buffer_id)
    {
        glGenBuffers(1 , &_buffer_id);
    }
}

void GLBuffer::finalize()
{
    if (0 != _buffer_id)
    {  
        glDeleteBuffers(1 , &_buffer_id);
        _buffer_id = 0;
    }
}

unsigned int GLBuffer::get_id() const
{
    return _buffer_id;
}

void GLBuffer::bind()
{
    if (0 == _buffer_id)
    {
        return;
        //GLRESOURCE_THROW_EXCEPTION("Invalid buffer!");
    }
    glBindBuffer(_target , _buffer_id);
}

void GLBuffer::unbind()
{
    glBindBuffer(_target , 0);
}

void GLBuffer::load(GLsizei size, const void* data, GLenum usage)
{
    glBufferData(_target , size , data, usage);
}

void GLBuffer::bind_buffer_base(GLenum target , GLuint index)
{
    glBindBufferBase(target , index, _buffer_id);
}



