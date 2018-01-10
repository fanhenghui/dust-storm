#ifndef MEDIMGRESOURCE_MI_GL_TEXTURE_1D_H
#define MEDIMGRESOURCE_MI_GL_TEXTURE_1D_H

#include "glresource/mi_gl_texture_base.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLTexture1D : public GLTextureBase {
public:
    explicit GLTexture1D(UIDType uid);

    ~GLTexture1D();

    virtual void bind();

    virtual void unbind();

    virtual float memory_used() const;

    void load(GLint internalformat, GLsizei width, GLenum format, GLenum type,
              const void* data, GLint level = 0);

    void update(GLint xoffset, GLsizei width, GLenum format, GLenum type,
                const void* data, GLint level = 0);

    void download(GLenum format, GLenum type, void* buffer,
                  GLint level = 0) const;

    GLsizei get_width() const;

    GLenum get_format() const;

    GLenum get_data_type() const;

protected:
private:
    GLsizei _width;
    GLenum _format;
    GLenum _internal_format;
    GLenum _data_type;
};

MED_IMG_END_NAMESPACE

#endif