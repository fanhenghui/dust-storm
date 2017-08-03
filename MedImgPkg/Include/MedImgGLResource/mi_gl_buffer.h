#ifndef MED_IMG_GL_BUFFER_H_
#define MED_IMG_GL_BUFFER_H_

#include "MedImgGLResource/mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLBuffer : public GLObject
{
public:
    GLBuffer(UIDType uid);

    ~GLBuffer();

    void set_buffer_target(GLenum target);

    GLenum get_buffer_target() const;

    virtual void initialize();

    virtual void finalize();

    unsigned int get_id() const;

    void bind_buffer_base(GLenum target , GLuint index);

    void bind();

    void unbind();

    void load(GLsizei size, const void* data, GLenum usage);

    void download(GLsizei size ,void* data);

private:
    GLenum _target;
    unsigned int _buffer_id;
};

MED_IMG_END_NAMESPACE

#endif
