#ifndef MEDIMGRESOURCE_GL_UTILS_H_
#define MEDIMGRESOURCE_GL_UTILS_H_

#include <sstream>
#include "GL/glew.h"
#include "glresource/mi_gl_resource_export.h"
#include "io/mi_io_define.h"

MED_IMG_BEGIN_NAMESPACE

#define CHECK_GL_ERROR                                                         \
  if (GLUtils::get_check_gl_flag()) {                                          \
    GLenum err = glGetError();                                                 \
    std::stringstream ss;                                                      \
    switch (err) {                                                             \
    case GL_NO_ERROR: {                                                        \
      break;                                                                   \
    }                                                                          \
    case GL_INVALID_ENUM: {                                                    \
      ss << "OpenGL Error: GL_INVALID_ENUM ! "                          \
                << "In File: " << __FILE__  << "In File: " << __FILE__  << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__;  \
      GLUtils::log_gl_error(ss.str());                                         \
      break;                                                                   \
    }                                                                          \
    case GL_INVALID_VALUE: {                                                   \
      ss << "OpenGL Error: GL_INVALID_VALUE ! "                         \
                << "In File: " << __FILE__  << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__;  \
      GLUtils::log_gl_error(ss.str());                                         \
      break;                                                                   \
    }                                                                          \
    case GL_INVALID_OPERATION: {                                               \
      ss << "OpenGL Error: GL_INVALID_OPERATION ! "                     \
                << "In File: " << __FILE__  << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__;  \
      GLUtils::log_gl_error(ss.str());                                         \
      break;                                                                   \
    }                                                                          \
    case GL_INVALID_FRAMEBUFFER_OPERATION: {                                   \
      ss << "OpenGL Error: GL_INVALID_FRAMEBUFFER_OPERATION ! "         \
                << "In File: " << __FILE__  << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__;  \
      GLUtils::log_gl_error(ss.str());                                         \
      break;                                                                   \
    }                                                                          \
    case GL_OUT_OF_MEMORY: {                                                   \
      ss << "OpenGL Error: GL_OUT_OF_MEMORY ! "                         \
                << "In File: " << __FILE__  << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__;  \
      GLUtils::log_gl_error(ss.str());                                         \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

class GLResource_Export GLUtils {
public:
    static bool check_framebuffer_state();

    static void get_gray_texture_format(DataType data_type,
                                        GLenum& internal_format, GLenum& format,
                                        GLenum& type);

    static unsigned int get_byte_by_data_type(DataType data_type);

    static std::string get_gl_enum_description(GLenum e);

    static void set_check_gl_flag(bool flag);

    static bool get_check_gl_flag();

    static void set_pixel_pack_alignment(int i);

    static void log_gl_error(const std::string& err);

private:
    static bool _s_check_gl_flag;
};

class GLResource_Export DrawFBOStack {
public:
    DrawFBOStack();
    ~DrawFBOStack();

private:
    GLint _current_draw_fbo;
    GLint _current_draw_buffer_count;
    GLint _current_draw_buffer_array[8];
};

class GLResource_Export ReadFBOStack {
public:
    ReadFBOStack();
    ~ReadFBOStack();

private:
    GLint _current_read_fbo;
    GLint _current_read_buffer;
};

class GLResource_Export FBOStack {
public:
    FBOStack();
    ~FBOStack();

private:
    GLint _current_draw_fbo;
    GLint _current_draw_buffer_count;
    GLint _current_draw_buffer_array[8];
    GLint _current_read_fbo;
    GLint _current_read_buffer;
};

class GLResource_Export GLActiveTextureCounter {
public:
    GLActiveTextureCounter();
    ~GLActiveTextureCounter();

    int tick();

    void reset();

private:
    int _current_active_texture_id;
};

class GLResource_Export GLContextHelper {
public:
    static bool has_gl_context();
};

class GLResource_Export GLTextureUtils {
public:
    static void set_1d_wrap_s(GLint wrap_type);
    static void set_1d_array_wrap_s(GLint wrap_type);
    static void set_2d_wrap_s_t(GLint wrap_type);
    static void set_3d_wrap_s_t_r(GLint wrap_type);
    static void set_filter(GLenum texture_target, GLint filter_type);
};

MED_IMG_END_NAMESPACE

#endif
