#include "mi_gl_texture_3d.h"
#include "mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

GLTexture3D::GLTexture3D(UIDType uid)
    : GLTextureBase(uid, "GLTexture3D"), _width(0), _height(0), _depth(0), _format(GL_RGBA),
      _internal_format(GL_RGBA8), _data_type(GL_UNSIGNED_BYTE) {
}

GLTexture3D::~GLTexture3D() {}

void GLTexture3D::bind() {
    glBindTexture(GL_TEXTURE_3D, _texture_id);
}

void GLTexture3D::unbind() {
    glBindTexture(GL_TEXTURE_3D, 0);
}

float GLTexture3D::memory_used() const {
    return _texture_id == 0 ? 0.0f : (_width * _height * _depth * GLUtils::get_component_byte(_format, _data_type)) / 1024.0f;
}

void GLTexture3D::load(GLint internalformat, GLsizei width, GLsizei height,
                       GLsizei depth, GLenum format, GLenum type,
                       const void* data, GLint level /*= 0*/) {
    glTexImage3D(GL_TEXTURE_3D, level, internalformat, width, height, depth, 0,
                 format, type, data);
    _width = width;
    _height = height;
    _depth = depth;
    _format = format;
    _internal_format = internalformat;
    _data_type = type;
}

void GLTexture3D::update(GLint xoffset, GLint yoffset, GLint zoffset,
                         GLsizei width, GLsizei height, GLsizei depth,
                         GLenum format, GLenum type, const void* data,
                         GLint level /*= 0*/) {
    glTexSubImage3D(GL_TEXTURE_3D, level, xoffset, yoffset, zoffset, width,
                    height, depth, format, type, data);
}

void GLTexture3D::download(GLenum format, GLenum type, void* buffer,
                           GLint level /*= 0*/) const {
    glGetTexImage(GL_TEXTURE_3D, level, format, type, buffer);
}

GLsizei GLTexture3D::get_width() const {
    return _width;
}

GLsizei GLTexture3D::get_height() const {
    return _height;
}

GLsizei GLTexture3D::get_depth() const {
    return _depth;
}

GLenum GLTexture3D::get_format() const {
    return _format;
}

GLenum GLTexture3D::get_data_type() const {
    return _data_type;
}

MED_IMG_END_NAMESPACE