#include "mi_gl_resource_manager.h"

#include "mi_gl_buffer.h"
#include "mi_gl_texture_1d.h"
#include "mi_gl_texture_1d_array.h"
#include "mi_gl_texture_2d.h"
#include "mi_gl_texture_3d.h"
#include "mi_gl_program.h"
#include "mi_gl_fbo.h"
#include "mi_gl_vao.h"
#include "mi_gl_context.h"
#include "mi_gl_time_query.h"

MED_IMG_BEGIN_NAMESPACE

template <>
std::string GLResourceManager<GLBuffer>::get_type() const {
    return "GLBuffer";
}

template <>
std::string GLResourceManager<GLTexture1D>::get_type() const {
    return "GLTexture1D";
}

template <>
std::string GLResourceManager<GLTexture2D>::get_type() const {
    return "GLTexture2D";
}

template <>
std::string GLResourceManager<GLTexture3D>::get_type() const {
    return "GLTexture3D";
}

template <>
std::string GLResourceManager<GLTexture1DArray>::get_type() const {
    return "GLTexture1DArray";
}

template <>
std::string GLResourceManager<GLProgram>::get_type() const {
    return "GLProgram";
}

template <>
std::string GLResourceManager<GLFBO>::get_type() const {
    return "GLFBO";
}

template <>
std::string GLResourceManager<GLVAO>::get_type() const {
    return "GLVAO";
}

template <>
std::string GLResourceManager<GLContext>::get_type() const {
    return "GLContext";
}

template <>
std::string GLResourceManager<GLTimeQuery>::get_type() const {
    return "GLTimeQuery";
}

MED_IMG_END_NAMESPACE