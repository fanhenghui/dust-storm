#include "mi_gl_texture_1d_array.h"
#include "mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

GLTexture1DArray::GLTexture1DArray(UIDType uid)
    : GLTextureBase(uid, "GLTexture1DArray"), _width(0), _array_size(0), _format(GL_RGBA),
      _internal_format(GL_RGBA8), _data_type(GL_UNSIGNED_BYTE) {
}

GLTexture1DArray::~GLTexture1DArray() {}

void GLTexture1DArray::bind() {
    glBindTexture(GL_TEXTURE_1D_ARRAY, _texture_id);
}

void GLTexture1DArray::unbind() {
    glBindTexture(GL_TEXTURE_1D_ARRAY, 0);
}

float GLTexture1DArray::memory_used() const {
    return _texture_id == 0 ? 0.0f : (_width * _array_size *  GLUtils::get_component_byte(_format, _data_type)) / 1024.0f;
}

void GLTexture1DArray::load(GLint internalformat, GLsizei width,
                            GLsizei arraysize, GLenum format, GLenum type,
                            const void* data, GLint level /*= 0*/) {
    glTexImage2D(GL_TEXTURE_1D_ARRAY, level, internalformat, width, arraysize, 0,
                 format, type, data);
    _width = width;
    _array_size = arraysize;
    _format = format;
    _internal_format = internalformat;
    _data_type = type;
}

void GLTexture1DArray::update(GLint xoffset, GLsizei arrayidx, GLsizei width,
                              GLenum format, GLenum type, const void* data,
                              GLint level /*= 0*/) {
    glTexSubImage2D(GL_TEXTURE_1D_ARRAY, level, xoffset, arrayidx, width, 1,
                    format, type, data);
}

void GLTexture1DArray::download(GLenum format, GLenum type, void* buffer,
                                GLint level /*= 0*/) const {
    glGetTexImage(GL_TEXTURE_1D_ARRAY, level, format, type, buffer);
}

int GLTexture1DArray::get_width() const {
    return _width;
}

int GLTexture1DArray::get_array_size() const {
    return _array_size;
}

GLenum GLTexture1DArray::get_format() const {
    return _format;
}

GLenum GLTexture1DArray::get_data_type() const {
    return _data_type;
}

MED_IMG_END_NAMESPACE