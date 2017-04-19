#ifndef MED_IMAGING_VAO_H
#define MED_IMAGING_VAO_H

#include "MedImgGLResource/mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLVAO : public GLObject
{
public:
    GLVAO(UIDType uid);

    ~GLVAO();

    virtual void Initialize();

    virtual void Finalize();

    unsigned int GetID() const;

    void Bind();

    void UnBind();

protected:
private:
    unsigned int m_uiVAOID;
};

MED_IMAGING_END_NAMESPACE

#endif