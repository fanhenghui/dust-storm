#ifndef MEDIMGRESOURCE_MI_GL_BUFFER_H
#define MEDIMGRESOURCE_MI_GL_BUFFER_H

#include "glresource/mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLBuffer : public GLObject {
public:
    explicit GLBuffer(UIDType uid);

    ~GLBuffer();

    void set_buffer_target(GLenum target);

    GLenum get_buffer_target() const;

    virtual void initialize();

    virtual void finalize();

    virtual float memory_used() const;

    unsigned int get_id() const;

    void bind_buffer_base(GLenum target, GLuint index);

    void bind();

    void unbind();

    void load(GLsizei size, const void* data, GLenum usage);

    void download(GLsizei size, void* data);

    size_t get_size() const;

private:
    GLenum _target;
    unsigned int _buffer_id;
    size_t _size;
};

MED_IMG_END_NAMESPACE

#endif
