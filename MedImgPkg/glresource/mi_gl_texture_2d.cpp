#include "mi_gl_texture_2d.h"
#include "mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

GLTexture2D::GLTexture2D(UIDType uid)
    : GLTextureBase(uid, "GLTexture2D"), _width(0), _height(0), _format(GL_RGBA),
      _internal_format(GL_RGBA8), _data_type(GL_UNSIGNED_BYTE) {
}

GLTexture2D::~GLTexture2D() {}

void GLTexture2D::bind() {
    glBindTexture(GL_TEXTURE_2D, _texture_id);
}

void GLTexture2D::unbind() {
    glBindTexture(GL_TEXTURE_2D, 0);
}

float GLTexture2D::memory_used() const {
    return _texture_id == 0 ? 0.0f : (_width * _height *  GLUtils::get_component_byte(_format, _data_type)) / 1024.0f;
}

void GLTexture2D::load(GLint internalformat, GLsizei width, GLsizei height,
                       GLenum format, GLenum type, const void* data,
                       GLint level /*= 0*/) {
    glTexImage2D(GL_TEXTURE_2D, level, internalformat, width, height, 0, format,
                 type, data);
    _width = width;
    _height = height;
    _format = format;
    _internal_format = internalformat;
    _data_type = type;
}

void GLTexture2D::update(GLint xoffset, GLint yoffset, GLsizei width,
                         GLsizei height, GLenum format, GLenum type,
                         const void* data, GLint level /*= 0*/) {
    glTexSubImage2D(GL_TEXTURE_2D, level, xoffset, yoffset, width, height, format,
                    type, data);
}

void GLTexture2D::download(GLenum format, GLenum type, void* buffer,
                           GLint level /*= 0*/) const {
    GLRESOURCE_CHECK_NULL_EXCEPTION(buffer);
    glGetTexImage(GL_TEXTURE_2D, level, format, type, buffer);
}

void GLTexture2D::read_pixels(GLenum format, GLenum type, GLint x, GLint y, 
                 GLsizei width, GLsizei height, void* pixels) {
    GLRESOURCE_CHECK_NULL_EXCEPTION(pixels);
    glReadPixels(x, y, width, height, format, type, pixels);
}

GLsizei GLTexture2D::get_width() const {
    return _width;
}

GLsizei GLTexture2D::get_height() const {
    return _height;
}

GLenum GLTexture2D::get_format() const {
    return _format;
}

GLenum GLTexture2D::get_data_type() const {
    return _data_type;
}

MED_IMG_END_NAMESPACE