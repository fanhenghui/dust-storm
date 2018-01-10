#include "mi_gl_buffer.h"

MED_IMG_BEGIN_NAMESPACE

GLBuffer::GLBuffer(UIDType uid)
    : GLObject(uid, "GLBuffer"), _target(GL_ARRAY_BUFFER), _buffer_id(0), _size(0) {
}

GLBuffer::~GLBuffer() {}

void GLBuffer::set_buffer_target(GLenum target) {
    _target = target;
}

GLenum GLBuffer::get_buffer_target() const {
    return _target;
}

void GLBuffer::initialize() {
    if (0 == _buffer_id) {
        glGenBuffers(1, &_buffer_id);
    }
}

void GLBuffer::finalize() {
    if (0 != _buffer_id) {
        glDeleteBuffers(1, &_buffer_id);
        _buffer_id = 0;
    }
}

float GLBuffer::memory_used() const {
    return 0 == _buffer_id ? 0.0f : _size/1024.0f;
}

unsigned int GLBuffer::get_id() const {
    return _buffer_id;
}

void GLBuffer::bind() {
    if (0 == _buffer_id) {
        GLRESOURCE_THROW_EXCEPTION("Invalid buffer!");
    }

    glBindBuffer(_target, _buffer_id);
}

void GLBuffer::unbind() {
    glBindBuffer(_target, 0);
}

void GLBuffer::load(GLsizei size, const void* data, GLenum usage) {
    _size = size;
    glBufferData(_target, size, data, usage);
}

void GLBuffer::download(GLsizei size, void* data) {
    void* buffer_data = glMapBuffer(_target, GL_READ_ONLY);
    memcpy(data, buffer_data, size);
    glUnmapBuffer(_target);
}

size_t GLBuffer::get_size() const {   
    return _buffer_id == 0 ? 0 : _size;
}

void GLBuffer::bind_buffer_base(GLenum target, GLuint index) {
    glBindBufferBase(target, index, _buffer_id);
}

MED_IMG_END_NAMESPACE