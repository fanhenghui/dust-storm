#ifndef MED_IMG_VAO_H
#define MED_IMG_VAO_H

#include "glresource/mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

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

MED_IMG_END_NAMESPACE

#endif