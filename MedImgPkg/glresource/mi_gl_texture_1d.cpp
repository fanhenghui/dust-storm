#include "mi_gl_texture_1d.h"
#include "mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

GLTexture1D::GLTexture1D(UIDType uid)
    : GLTextureBase(uid, "GLTexture1D"), _width(0), _format(GL_RGBA),
      _internal_format(GL_RGBA8), _data_type(GL_UNSIGNED_BYTE) {
}

GLTexture1D::~GLTexture1D() {}

void GLTexture1D::bind() {
    glBindTexture(GL_TEXTURE_1D, _texture_id);
}

void GLTexture1D::unbind() {
    glBindTexture(GL_TEXTURE_1D, 0);
}

float GLTexture1D::memory_used() const {
    return _texture_id == 0 ? 0.0f : (_width * GLUtils::get_component_byte(_format, _data_type))/1024.0f;
}

void GLTexture1D::load(GLint internalformat, GLsizei width, GLenum format,
                       GLenum type, const void* data, GLint level /*= 0*/) {
    glTexImage1D(GL_TEXTURE_1D, level, internalformat, width, 0, format, type, data);
    _width = width;
    _format = format;
    _internal_format = internalformat;
    _data_type = type;
}

void GLTexture1D::update(GLint xoffset, GLsizei width, GLenum format,
                         GLenum type, const void* data, GLint level /*= 0*/) {
    glTexSubImage1D(GL_TEXTURE_1D, level, xoffset, width, format, type, data);
}

void GLTexture1D::download(GLenum format, GLenum type, void* buffer,
                           GLint level /*= 0*/) const {
    glGetTexImage(GL_TEXTURE_1D, level, format, type, buffer);
}

GLsizei GLTexture1D::get_width() const {
    return _width;
}

GLenum GLTexture1D::get_format() const {
    return _format;
}

GLenum GLTexture1D::get_data_type() const {
    return _data_type;
}

MED_IMG_END_NAMESPACE