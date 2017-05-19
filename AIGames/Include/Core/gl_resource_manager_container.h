#ifndef MED_IMAGING_GL_RESOURCE_MANAGER_CONTAINER_H
#define MED_IMAGING_GL_RESOURCE_MANAGER_CONTAINER_H

#include "Core/gl_resource_manager.h"



class GLProgram;
class GLBuffer;
class GLTexture1D;
class GLTexture2D;
class GLTexture3D;
class GLVAO;
class GLFBO;
class GLTexture1DArray;

typedef GLResourceManager<GLProgram> GLProgramManager;
typedef GLResourceManager<GLBuffer> GLBufferManager;
typedef GLResourceManager<GLTexture1D> GLTexture1DManager;
typedef GLResourceManager<GLTexture2D> GLTexture2DManager;
typedef GLResourceManager<GLTexture3D> GLTexture3DManager;
typedef GLResourceManager<GLVAO> GLVAOManager;
typedef GLResourceManager<GLFBO> GLFBOManager;
typedef GLResourceManager<GLTexture1DArray> GLTexture1DArrayManager;


typedef std::shared_ptr<GLProgramManager> GLProgramManagerPtr;
typedef std::shared_ptr<GLBufferManager> GLBufferManagerPtr;
typedef std::shared_ptr<GLTexture1DManager> GLTexture1DManagerPtr;
typedef std::shared_ptr<GLTexture2DManager> GLTexture2DManagerPtr;
typedef std::shared_ptr<GLTexture3DManager> GLTexture3DManagerPtr;
typedef std::shared_ptr<GLVAOManager> GLVAOManagerPtr;
typedef std::shared_ptr<GLFBOManager> GLFBOManagerPtr;
typedef std::shared_ptr<GLTexture1DArrayManager> GLTexture1DArrayManagerPtr;

class  GLResourceManagerContainer
{
public:
    static GLResourceManagerContainer* instance();

    ~GLResourceManagerContainer();

    GLProgramManagerPtr get_program_manager() const;

    GLBufferManagerPtr get_buffer_manager() const;

    GLTexture1DManagerPtr get_texture_1d_manager() const;

    GLTexture2DManagerPtr get_texture_2d_manager() const;

    GLTexture3DManagerPtr get_texture_3d_manager() const;

    GLTexture1DArrayManagerPtr get_texture_1d_array_manager() const;

    GLVAOManagerPtr get_vao_manager() const;

    GLFBOManagerPtr get_fbo_manager() const;

    void update_all();

private:
    GLResourceManagerContainer();
private:
    static GLResourceManagerContainer* _s_instance;
    static std::mutex _mutex;
private:
    GLProgramManagerPtr _program_manager;
    GLBufferManagerPtr _buffer_manager;
    GLTexture1DManagerPtr _texture_1d_manager;
    GLTexture2DManagerPtr _texture_2d_manager;
    GLTexture3DManagerPtr _texture_3d_manager;
    GLTexture1DArrayManagerPtr _texture_1d_array_manager;
    GLVAOManagerPtr _vao_manager;
    GLFBOManagerPtr _fbo_manager;
};




#endif
