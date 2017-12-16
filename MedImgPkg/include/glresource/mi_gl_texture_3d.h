#ifndef MEDIMGRESOURCE_TEXTURE_3D_H
#define MEDIMGRESOURCE_TEXTURE_3D_H

#include "glresource/mi_gl_texture_base.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLTexture3D : public GLTextureBase {
public:
    explicit GLTexture3D(UIDType uid);

    ~GLTexture3D();

    virtual void bind();

    virtual void unbind();

    void load(GLint internalformat, GLsizei width, GLsizei height, GLsizei depth,
              GLenum format, GLenum type, const void* data, GLint level = 0);

    void update(GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                GLsizei height, GLsizei depth, GLenum format, GLenum type,
                const void* data, GLint level = 0);

    void download(GLenum format, GLenum type, void* buffer,
                  GLint level = 0) const;

    GLsizei get_width();

    GLsizei get_height();

    GLsizei get_depth();

    GLenum get_format();

protected:
private:
    GLsizei _width;
    GLsizei _height;
    GLsizei _depth;
    GLenum _format;
    GLenum _internal_format;
    GLenum _type;
};

MED_IMG_END_NAMESPACE

#endif