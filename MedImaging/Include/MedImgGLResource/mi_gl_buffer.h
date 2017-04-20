#ifndef MED_IMAGING_GL_BUFFER_H_
#define MED_IMAGING_GL_BUFFER_H_

#include "MedImgGLResource/mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

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

    void bind();

    void unbind();

    void load(GLsizei size, const void* data, GLenum usage);

private:
    GLenum m_eTarget;
    unsigned int m_uiBufferID;
};

MED_IMAGING_END_NAMESPACE

#endif
