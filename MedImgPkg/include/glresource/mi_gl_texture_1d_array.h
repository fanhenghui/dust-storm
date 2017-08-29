#ifndef MEDIMGRESOURCE_TEXTURE_1D_ARRAY_H
#define MEDIMGRESOURCE_TEXTURE_1D_ARRAY_H

#include "glresource/mi_gl_texture_base.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLTexture1DArray : public GLTextureBase {
public:
    GLTexture1DArray(UIDType uid);

    ~GLTexture1DArray();

    virtual void bind();

    virtual void unbind();

    void load(GLint internalformat, GLsizei width, GLsizei arraysize,
              GLenum format, GLenum type, const void* data, GLint level = 0);

    void update(GLint xoffset, GLsizei arrayidx, GLsizei width, GLenum format,
                GLenum type, const void* data, GLint level = 0);

    void download(GLenum format, GLenum type, void* buffer,
                  GLint level = 0) const;

    int get_width();

    int get_array_size();

    GLenum get_format();

protected:
private:
    GLsizei _width;
    GLsizei _array_size;
    GLenum _format;
    GLenum _internal_format;
    GLenum _type;
};

MED_IMG_END_NAMESPACE

#endif