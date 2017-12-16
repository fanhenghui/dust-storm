#ifndef MEDIMGRESOURCE_FBO_H
#define MEDIMGRESOURCE_FBO_H

#include "glresource/mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

class GLTexture2D;
class GLResource_Export GLFBO : public GLObject {
public:
    explicit GLFBO(UIDType uid);

    ~GLFBO();

    virtual void initialize();

    virtual void finalize();

    unsigned int get_id() const;

    void bind();

    void unbind();

    void set_target(GLenum target);

    GLenum get_target();

    void attach_texture(GLenum attachment, std::shared_ptr<GLTexture2D> tex);

protected:
private:
    unsigned int _fbo_id;
    GLenum _target;
};

MED_IMG_END_NAMESPACE

#endif