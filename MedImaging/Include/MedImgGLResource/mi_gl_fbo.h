#ifndef MED_IMAGING_FBO_H
#define MED_IMAGING_FBO_H

#include "MedImgGLResource/mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLTexture2D;
class GLResource_Export GLFBO : public GLObject
{
public:
    GLFBO(UIDType uid);

    ~GLFBO();

    virtual void Initialize();

    virtual void Finalize();

    unsigned int GetID() const;

    void Bind();

    void UnBind();

    void SetTraget(GLenum eTarget);

    GLenum GetTraget();

    void AttachTexture(GLenum eAttach , std::shared_ptr<GLTexture2D> pTex2D);

protected:
private:
    unsigned int m_uiFBOID;
    GLenum m_eTarget;
};

MED_IMAGING_END_NAMESPACE

#endif