#ifndef MED_IMAGING_GL_UTILS_H_
#define MED_IMAGING_GL_UTILS_H_

#include "GL\glew.h"
#include "MedImgGLResource\mi_gl_resource_stdafx.h"
#include "MedImgCommon\mi_common_define.h"

MED_IMAGING_BEGIN_NAMESPACE

#define CHECK_GL_ERROR \
if(GLUtils::GetCheckGLFlag())\
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
    static bool CheckFramebufferState();

    static void GetGrayTextureFormat(DataType eDataType , GLenum &eInternalFormat , GLenum &eFormat , GLenum &eType );

    static unsigned int GetByteByDataType(DataType eDataType);

    static std::string GetGLenumDescription(GLenum e);

    static void SetCheckGLFlag(bool bFlag);

    static bool GetCheckGLFlag();

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

    int Tick();

    void Reset();

private:
    int m_iCurActiveTexID;
};

class GLResource_Export GLContextHelper
{
public:
    static bool HasGLContext();
};

class GLResource_Export GLTextureUtils
{
public:
    static void Set1DWrapS(GLint iWrapType);
    static void Set2DWrapST(GLint iWrapType);
    static void Set3DWrapSTR(GLint iWrapType);
    static void SetFilter(GLenum eTexTarget , GLint iFilterType);
};


MED_IMAGING_END_NAMESPACE

#endif
