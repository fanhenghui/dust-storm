#ifndef MED_IMAGING_GL_UTILS_H_
#define MED_IMAGING_GL_UTILS_H_

#include "GL\glew.h"
#include "MedImgGLResource\mi_gl_resource_stdafx.h"
#include "MedImgCommon\mi_common_define.h"

MED_IMAGING_BEGIN_NAMESPACE

#define CHECK_GL_ERROR \
if(GLUtils::get_check_gl_flag())\
{\
    GLenum err = glGetError();\
    switch(err)\
{\
    case GL_NO_ERROR:\
{\
    break;\
}\
    case GL_INVALID_ENUM:\
{\
    std::cout << "OpenGL Error: GL_INVALID_ENUM ! " << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__ << " .\n";\
    break;\
}\
    case GL_INVALID_VALUE:\
{\
    std::cout << "OpenGL Error: GL_INVALID_VALUE ! " << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__ << " .\n";\
    break;\
}\
    case GL_INVALID_OPERATION:\
{\
    std::cout << "OpenGL Error: GL_INVALID_OPERATION ! " << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__ << " .\n";\
    break;\
}\
    case GL_INVALID_FRAMEBUFFER_OPERATION:\
{\
    std::cout << "OpenGL Error: GL_INVALID_FRAMEBUFFER_OPERATION ! " << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__ << " .\n";\
    break;\
}\
    case GL_OUT_OF_MEMORY:\
{\
    std::cout << "OpenGL Error: GL_OUT_OF_MEMORY ! " << " In Line : " << __LINE__ << ". In Func " << __FUNCTION__ << " .\n";\
    break;\
}\
    default:\
    break;\
}\
}\


class GLResource_Export GLUtils
{
public:
    static bool check_framebuffer_state();

    static void get_gray_texture_format(DataType eDataType , GLenum &eInternalFormat , GLenum &eFormat , GLenum &eType );

    static unsigned int get_byte_by_data_type(DataType eDataType);

    static std::string get_gl_enum_description(GLenum e);

    static void set_check_gl_flag(bool bFlag);

    static bool get_check_gl_flag();

private:
    static bool m_bCheckGLFlag;
};

class GLResource_Export DrawFBOStack
{
public:
    DrawFBOStack();
    ~DrawFBOStack();
private:
    GLint m_iCurDrawFBO;
    GLint m_iCurDrawBufferCount;
    GLint m_iCurDrawBufferArray[8];
};

class GLResource_Export ReadFBOStack
{
public:
    ReadFBOStack();
    ~ReadFBOStack();
private:
    GLint m_iCurReadFBO;
    GLint m_iCurReadBuffer;
};

class GLResource_Export FBOStack
{
public:
    FBOStack();
    ~FBOStack();
private:
    GLint m_iCurDrawFBO;
    GLint m_iCurDrawBufferCount;
    GLint m_iCurDrawBufferArray[8];
    GLint m_iCurReadFBO;
    GLint m_iCurReadBuffer;
};

class GLResource_Export GLActiveTextureCounter
{
public:
    GLActiveTextureCounter();

    ~GLActiveTextureCounter();

    int tick();

    void reset();

private:
    int m_iCurActiveTexID;
};

class GLResource_Export GLContextHelper
{
public:
    static bool has_gl_context();
};

class GLResource_Export GLTextureUtils
{
public:
    static void set_1d_wrap_s(GLint iWrapType);
    static void set_2d_wrap_s_t(GLint iWrapType);
    static void set_1d_wrap_s_t_r(GLint iWrapType);
    static void set_filter(GLenum eTexTarget , GLint iFilterType);
};


MED_IMAGING_END_NAMESPACE

#endif
