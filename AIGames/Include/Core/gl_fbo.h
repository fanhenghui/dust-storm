#ifndef MED_IMAGING_FBO_H
#define MED_IMAGING_FBO_H

#include "Core/gl_object.h"



class GLTexture2D;
class  GLFBO : public GLObject
{
public:
    GLFBO(UIDType uid);

    ~GLFBO();

    virtual void initialize();

    virtual void finalize();

    unsigned int get_id() const;

    void bind();

    void unbind();

    void set_target(GLenum target);

    GLenum get_target();

    void attach_texture(GLenum attachment , std::shared_ptr<GLTexture2D> tex);

protected:
private:
    unsigned int _fbo_id;
    GLenum _target;
};



#endif