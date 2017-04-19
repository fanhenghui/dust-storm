#ifndef MED_IMAGING_GL_BUFFER_H_
#define MED_IMAGING_GL_BUFFER_H_

#include "MedImgGLResource/mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLBuffer : public GLObject
{
public:
    GLBuffer(UIDType uid);

    ~GLBuffer();

    void SetBufferTarget(GLenum target);

    GLenum GetBufferTarget() const;

    virtual void Initialize();

    virtual void Finalize();

    unsigned int GetID() const;

    void Bind();

    void UnBind();

    void Load(GLsizei size, const void* data, GLenum usage);

private:
    GLenum m_eTarget;
    unsigned int m_uiBufferID;
};

MED_IMAGING_END_NAMESPACE

#endif
