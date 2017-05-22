#include "gl_utils.h"

#ifdef WIN32
#include <Windows.h>
#endif



bool GLUtils::check_framebuffer_state()
{
    bool fbo_status_complete = false;
    GLenum fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (fbo_status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        {
            fbo_status_complete = true;
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

    return fbo_status_complete;
}

std::string GLUtils::get_gl_enum_description(GLenum e)
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

bool GLUtils::_s_check_gl_flag = true;
void GLUtils::set_check_gl_flag(bool flag)
{
    _s_check_gl_flag = flag;
}

bool GLUtils::get_check_gl_flag()
{
    return _s_check_gl_flag;
}

DrawFBOStack::DrawFBOStack()
{
    //Push draw frame buffer
    _current_draw_fbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING , &_current_draw_fbo);

    _current_draw_buffer_count = 0;
    memset(_current_draw_buffer_array , 0 , sizeof(int)*8 );

    for (int i = 0 ; i< 8 ; ++i)
    {
        glGetIntegerv(GL_DRAW_BUFFER0+i , (_current_draw_buffer_array + i));
        if (0 == _current_draw_buffer_array[i])
        {
            _current_draw_buffer_count = i;
            break;
        }
    }

    CHECK_GL_ERROR;
}

DrawFBOStack::~DrawFBOStack()
{
    //Pop draw frame buffer
    GLenum draw_buffer_array[8];
    for (int i = 0 ; i<8 ; ++i)
    {
        draw_buffer_array[i] = (GLenum)_current_draw_buffer_array[i];
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER , _current_draw_fbo);
    glDrawBuffers(_current_draw_buffer_count , draw_buffer_array);


    CHECK_GL_ERROR;
}


ReadFBOStack::ReadFBOStack()
{
    //Push read frame buffer
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING , &_current_read_fbo);
    glGetIntegerv(GL_READ_BUFFER,  &_current_read_buffer);


    CHECK_GL_ERROR;
}

ReadFBOStack::~ReadFBOStack()
{
    //Pop read frame buffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER , _current_read_fbo);
    glReadBuffer((GLenum)_current_read_buffer);


    CHECK_GL_ERROR;
}

FBOStack::FBOStack()
{
    //Push draw frame buffer
    _current_draw_fbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING , &_current_draw_fbo);

    _current_draw_buffer_count = 0;
    memset(_current_draw_buffer_array , 0 , sizeof(int)*8 );

    for (int i = 0 ; i< 8 ; ++i)
    {
        glGetIntegerv(GL_DRAW_BUFFER0+i , (_current_draw_buffer_array + i));
        if (0 == _current_draw_buffer_array[i])
        {
            _current_draw_buffer_count = i;
            break;
        }
    }

    //Push read frame buffer
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING , &_current_read_fbo);
    glGetIntegerv(GL_READ_BUFFER,  &_current_read_buffer);


    CHECK_GL_ERROR;
}

FBOStack::~FBOStack()
{
    //Pop draw frame buffer
    GLenum draw_buffer_array[8];
    for (int i = 0 ; i<8 ; ++i)
    {
        draw_buffer_array[i] = (GLenum)_current_draw_buffer_array[i];
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER , _current_draw_fbo);
    glDrawBuffers(_current_draw_buffer_count , draw_buffer_array);

    //Pop read frame buffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER , _current_read_fbo);
    glReadBuffer((GLenum)_current_read_buffer);


    CHECK_GL_ERROR;
}

GLActiveTextureCounter::GLActiveTextureCounter():_current_active_texture_id(0)
{}

GLActiveTextureCounter::~GLActiveTextureCounter()
{}

int GLActiveTextureCounter::tick()
{
    if (_current_active_texture_id > GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS  - 1)
    {
        throw std::exception("Combined texture count is beyond limitation!");
    }
    return _current_active_texture_id++;
}

void GLActiveTextureCounter::reset()
{
    _current_active_texture_id = 0;
}

bool GLContextHelper::has_gl_context()
{
    //TODO WR here is not cross-platform. Should add other OS define
#ifdef WIN32
    HDC hdc = wglGetCurrentDC();
    return (hdc != NULL);
#else
    return true;
#endif
}

void GLTextureUtils::set_1d_wrap_s( GLint wrap_type )
{
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, wrap_type);
    if (GL_CLAMP_TO_BORDER == wrap_type)
    {
        const float board[4] = {0.0f,0.0f,0.0f,0.0f};
        glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR , board);
    }
}

void GLTextureUtils::set_2d_wrap_s_t( GLint wrap_type )
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_type); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_type); 
    if (GL_CLAMP_TO_BORDER == wrap_type)
    {
        const float board[4] = {0.0f,0.0f,0.0f,0.0f};
        glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR , board);
    }
}

void GLTextureUtils::set_3d_wrap_s_t_r( GLint wrap_type )
{
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap_type); 
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap_type); 
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap_type); 
    if (GL_CLAMP_TO_BORDER == wrap_type)
    {
        const float board[4] = {0.0f,0.0f,0.0f,0.0f};
        glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR , board);
    }
}

void GLTextureUtils::set_filter( GLenum texture_target , GLint filter_type )
{
    glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, filter_type); 
    glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, filter_type); 
}



