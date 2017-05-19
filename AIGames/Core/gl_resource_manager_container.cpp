#include "gl_resource_manager_container.h"
#include "gl_program.h"
#include "gl_buffer.h"



    std::mutex GLResourceManagerContainer::_mutex;

GLResourceManagerContainer* GLResourceManagerContainer::_s_instance = nullptr;

GLResourceManagerContainer* GLResourceManagerContainer::instance()
{
    if (nullptr == _s_instance)
    {
        std::unique_lock<std::mutex> locker(_mutex);
        if (nullptr == _s_instance)
        {
            _s_instance = new GLResourceManagerContainer();
        }
    }
    return _s_instance;
}

GLResourceManagerContainer::~GLResourceManagerContainer()
{

}

GLProgramManagerPtr GLResourceManagerContainer::get_program_manager() const
{
    return _program_manager;
}

GLBufferManagerPtr GLResourceManagerContainer::get_buffer_manager() const
{
    return _buffer_manager;
}

void GLResourceManagerContainer::update_all()
{
    _program_manager->update();
    _buffer_manager->update();
}

GLResourceManagerContainer::GLResourceManagerContainer():
_program_manager(new GLProgramManager()),
    _buffer_manager(new GLBufferManager()),
    _texture_1d_manager(new GLTexture1DManager()),
    _texture_2d_manager(new GLTexture2DManager()),
    _texture_3d_manager(new GLTexture3DManager()),
    _vao_manager(new GLVAOManager()),
    _fbo_manager(new GLFBOManager()),
    _texture_1d_array_manager(new GLTexture1DArrayManager())
{

}

GLTexture1DManagerPtr GLResourceManagerContainer::get_texture_1d_manager() const
{
    return _texture_1d_manager;
}

GLTexture2DManagerPtr GLResourceManagerContainer::get_texture_2d_manager() const
{
    return _texture_2d_manager;
}

GLTexture3DManagerPtr GLResourceManagerContainer::get_texture_3d_manager() const
{
    return _texture_3d_manager;
}

GLVAOManagerPtr GLResourceManagerContainer::get_vao_manager() const
{
    return _vao_manager;
}

GLFBOManagerPtr GLResourceManagerContainer::get_fbo_manager() const
{
    return _fbo_manager;
}

GLTexture1DArrayManagerPtr GLResourceManagerContainer::get_texture_1d_array_manager() const
{
    return _texture_1d_array_manager;
}

