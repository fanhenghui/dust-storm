#ifndef MED_IMAGING_VAO_H
#define MED_IMAGING_VAO_H

#include "MedImgGLResource/mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLVAO : public GLObject
{
public:
    GLVAO(UIDType uid);

    ~GLVAO();

    virtual void initialize();

    virtual void finalize();

    unsigned int get_id() const;

    void bind();

    void unbind();

protected:
private:
    unsigned int _vao_id;
};

MED_IMAGING_END_NAMESPACE

#endif