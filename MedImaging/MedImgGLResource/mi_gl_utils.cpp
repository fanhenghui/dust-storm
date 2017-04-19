#include "mi_gl_utils.h"
#include "MedImgCommon/mi_common_exception.h"

#ifdef WIN32
#include <Windows.h>
#endif

MED_IMAGING_BEGIN_NAMESPACE

bool GLUtils::CheckFramebufferState()
{
    bool bFBOStatusComplete = false;
    GLenum eFBOStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (eFBOStatus)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        {
            bFBOStatusComplete = true;
            break;
        }
    case GL_FRAMEBUFFER_UNDEFINED:
        {
            std::cout << "Framebuffer undefined!\n";
            break;
        }
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        {
            std::cout << "Framebuffer incomplete attachment!\n";
            break;
        }
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        {
            std::cout << "Framebuffer incomplete missing attachment!\n";
            break;
        }
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        {
            std::cout << "Framebuffer incomplete draw buffer!\n";
            break;
        }
    case GL_FRAMEBUFFER_UNSUPPORTED:
        {
            std::cout << "Framebuffer unsupported!\n";
            break;
        }
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        {
            std::cout << "Framebuffer incomplete mulitisample!\n";
            break;
        }
    default:
        {
            std::cout << "Undefined framebuffer status!\n";
            break;
        }
    }

    return bFBOStatusComplete;
}

void GLUtils::GetGrayTextureFormat( DataType eDataType , GLenum &eInternalFormat , GLenum &eFormat , GLenum &eType )
{
    switch(eDataType)
    {
    case CHAR:
    case UCHAR:
        {
            eInternalFormat = GL_R8;
            eFormat = GL_RED;
            eType = GL_UNSIGNED_BYTE;
            break;
        }
    case SHORT:
    case USHORT:
        {
            eInternalFormat = GL_R16;
            eFormat = GL_RED;
            eType = GL_UNSIGNED_SHORT;
            break;
        }
    case FLOAT:
        {
            eInternalFormat = GL_R32F;
            eFormat = GL_RED;
            eType = GL_FLOAT;
            break;
        }
    default:
        GLRESOURCE_THROW_EXCEPTION("Unsupported data type!");
        break;
    }
}

unsigned int GLUtils::GetByteByDataType(DataType eDataType)
{
    switch(eDataType)
    {
    case CHAR:
    case UCHAR:
        {
            return 1;
            break;
        }
    case SHORT:
    case USHORT:
        {
            return 2;
            break;
        }
    case FLOAT:
        {
            return sizeof(float);
            break;
        }
    default:
        GLRESOURCE_THROW_EXCEPTION("Unsupported data type!");
        break;
    }
}

std::string GLUtils::GetGLenumDescription(GLenum e)
{
    switch(e)
    {
        //////////////////////////////////////////////////////////////////////////
        //Format
    case GL_R:
        return std::string("GL_R");
    case GL_RGB:
        return std::string("GL_RGB");
    case GL_RGBA:
        return std::string("GL_RGBA");
    case GL_LUMINANCE:
        return std::string("GL_LUMINANCE");
    case GL_R8:
        return std::string("GL_R8");
    case GL_RGB8:
        return std::string("GL_RGB8");
    case GL_RGBA8:
        return std::string("GL_RGBA8");
    case GL_LUMINANCE8:
        return std::string("GL_LUMINANCE8");
    case GL_R16:
        return std::string("GL_R16");
    case GL_RGB16:
        return std::string("GL_RGB16");
    case GL_RGBA16:
        return std::string("GL_RGBA16");
    case GL_LUMINANCE16:
        return std::string("GL_LUMINANCE16");
    case GL_R16F:
        return std::string("GL_R16F");
    case GL_R32F:
        return std::string("GL_R32F");
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //Buffer target
    case GL_ARRAY_BUFFER:
        return std::string("GL_ARRAY_BUFFER");
    case GL_SHADER_STORAGE_BUFFER:
        return std::string("GL_SHADER_STORAGE_BUFFER");
    case GL_ELEMENT_ARRAY_BUFFER:
        return std::string("GL_ELEMENT_ARRAY_BUFFER");
    case GL_READ_BUFFER:
        return std::string("GL_READ_BUFFER");
    case GL_DRAW_BUFFER:
        return std::string("GL_DRAW_BUFFER");
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //Frame buffer target
    case GL_FRAMEBUFFER:
        return std::string("GL_FRAMEBUFFER");
    case GL_READ_FRAMEBUFFER:
        return std::string("GL_READ_FRAMEBUFFER");
    case GL_DRAW_FRAMEBUFFER:
        return std::string("GL_DRAW_FRAMEBUFFER");
        //////////////////////////////////////////////////////////////////////////

    default:
        return "Undefined";
    }
}

bool GLUtils::m_bCheckGLFlag = true;
void GLUtils::SetCheckGLFlag(bool bFlag)
{
    m_bCheckGLFlag = bFlag;
}

bool GLUtils::GetCheckGLFlag()
{
    return m_bCheckGLFlag;
}

DrawFBOStack::DrawFBOStack()
{
    //Push draw frame buffer
    m_iCurDrawFBO = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING , &m_iCurDrawFBO);

    m_iCurDrawBufferCount = 0;
    memset(m_iCurDrawBufferArray , 0 , sizeof(int)*8 );

    for (int i = 0 ; i< 8 ; ++i)
    {
        glGetIntegerv(GL_DRAW_BUFFER0+i , (m_iCurDrawBufferArray + i));
        if (0 == m_iCurDrawBufferArray[i])
        {
            m_iCurDrawBufferCount = i;
            break;
        }
    }

    CHECK_GL_ERROR;
}

DrawFBOStack::~DrawFBOStack()
{
    //Pop draw frame buffer
    GLenum eDrawBufferArray[8];
    for (int i = 0 ; i<8 ; ++i)
    {
        eDrawBufferArray[i] = (GLenum)m_iCurDrawBufferArray[i];
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER , m_iCurDrawFBO);
    glDrawBuffers(m_iCurDrawBufferCount , eDrawBufferArray);


    CHECK_GL_ERROR;
}


ReadFBOStack::ReadFBOStack()
{
    //Push read frame buffer
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING , &m_iCurReadFBO);
    glGetIntegerv(GL_READ_BUFFER,  &m_iCurReadBuffer);


    CHECK_GL_ERROR;
}

ReadFBOStack::~ReadFBOStack()
{
    //Pop read frame buffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER , m_iCurReadFBO);
    glReadBuffer((GLenum)m_iCurReadBuffer);


    CHECK_GL_ERROR;
}

FBOStack::FBOStack()
{
    //Push draw frame buffer
    m_iCurDrawFBO = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING , &m_iCurDrawFBO);

    m_iCurDrawBufferCount = 0;
    memset(m_iCurDrawBufferArray , 0 , sizeof(int)*8 );

    for (int i = 0 ; i< 8 ; ++i)
    {
        glGetIntegerv(GL_DRAW_BUFFER0+i , (m_iCurDrawBufferArray + i));
        if (0 == m_iCurDrawBufferArray[i])
        {
            m_iCurDrawBufferCount = i;
            break;
        }
    }

    //Push read frame buffer
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING , &m_iCurReadFBO);
    glGetIntegerv(GL_READ_BUFFER,  &m_iCurReadBuffer);


    CHECK_GL_ERROR;
}

FBOStack::~FBOStack()
{
    //Pop draw frame buffer
    GLenum eDrawBufferArray[8];
    for (int i = 0 ; i<8 ; ++i)
    {
        eDrawBufferArray[i] = (GLenum)m_iCurDrawBufferArray[i];
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER , m_iCurDrawFBO);
    glDrawBuffers(m_iCurDrawBufferCount , eDrawBufferArray);

    //Pop read frame buffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER , m_iCurReadFBO);
    glReadBuffer((GLenum)m_iCurReadBuffer);


    CHECK_GL_ERROR;
}

GLActiveTextureCounter::GLActiveTextureCounter():m_iCurActiveTexID(0)
{}

GLActiveTextureCounter::~GLActiveTextureCounter()
{}

int GLActiveTextureCounter::Tick()
{
    if (m_iCurActiveTexID > GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS  - 1)
    {
        throw std::exception("Combined texture count is beyond limitation!");
    }
    return m_iCurActiveTexID++;
}

void GLActiveTextureCounter::Reset()
{
    m_iCurActiveTexID = 0;
}

bool GLContextHelper::HasGLContext()
{
    //TODO WR here is not cross-platform. Should add other OS define
#ifdef WIN32
    HDC hdc = wglGetCurrentDC();
    return (hdc != NULL);
#else
    return true;
#endif
}

void GLTextureUtils::Set1DWrapS( GLint iWrapType )
{
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, iWrapType);
    if (GL_CLAMP_TO_BORDER == iWrapType)
    {
        const float fBoard[4] = {0.0f,0.0f,0.0f,0.0f};
        glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR , fBoard);
    }
}

void GLTextureUtils::Set2DWrapST( GLint iWrapType )
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, iWrapType); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, iWrapType); 
    if (GL_CLAMP_TO_BORDER == iWrapType)
    {
        const float fBoard[4] = {0.0f,0.0f,0.0f,0.0f};
        glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR , fBoard);
    }
}

void GLTextureUtils::Set3DWrapSTR( GLint iWrapType )
{
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, iWrapType); 
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, iWrapType); 
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, iWrapType); 
    if (GL_CLAMP_TO_BORDER == iWrapType)
    {
        const float fBoard[4] = {0.0f,0.0f,0.0f,0.0f};
        glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR , fBoard);
    }
}

void GLTextureUtils::SetFilter( GLenum eTexTarget , GLint iFilterType )
{
    glTexParameteri(eTexTarget, GL_TEXTURE_MIN_FILTER, iFilterType); 
    glTexParameteri(eTexTarget, GL_TEXTURE_MAG_FILTER, iFilterType); 
}

MED_IMAGING_END_NAMESPACE

