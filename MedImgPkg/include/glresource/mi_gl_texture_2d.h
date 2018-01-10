#ifndef MEDIMGRESOURCE_MI_GL_TEXTURE_2D_H
#define MEDIMGRESOURCE_MI_GL_TEXTURE_2D_H

#include "glresource/mi_gl_texture_base.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLTexture2D : public GLTextureBase {
public:
    explicit GLTexture2D(UIDType uid);

    ~GLTexture2D();

    virtual void bind();

    virtual void unbind();

    virtual float memory_used() const;

    void load(GLint internalformat, GLsizei width, GLsizei height, GLenum format,
              GLenum type, const void* data, GLint level = 0);

    void update(GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                GLenum format, GLenum type, const void* data, GLint level = 0);

    void download(GLenum format, GLenum type, void* buffer,
                  GLint level = 0) const;

    void read_pixels(GLenum format, GLenum type, GLint x, GLint y, 
                     GLsizei width, GLsizei height, void* pixels);

    GLsizei get_width() const;

    GLsizei get_height() const;

    GLenum get_format() const;

    GLenum get_data_type() const;

protected:
private:
    GLsizei _width;
    GLsizei _height;
    GLenum _format;
    GLenum _internal_format;
    GLenum _data_type;
};

MED_IMG_END_NAMESPACE

#endif