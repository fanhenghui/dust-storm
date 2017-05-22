#ifndef MED_IMAGING_GL_BUFFER_H_
#define MED_IMAGING_GL_BUFFER_H_

#include "Core/gl_object.h"



class  GLBuffer : public GLObject
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

private:
    GLenum _target;
    unsigned int _buffer_id;
};



#endif
