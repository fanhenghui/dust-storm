#ifndef MED_IMAGING_VAO_H
#define MED_IMAGING_VAO_H

#include "Core/gl_object.h"



class  GLVAO : public GLObject
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



#endif