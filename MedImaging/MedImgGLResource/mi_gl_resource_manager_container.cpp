#include "mi_gl_resource_manager_container.h"
#include "mi_gl_program.h"
#include "mi_gl_buffer.h"

MED_IMAGING_BEGIN_NAMESPACE

    boost::mutex GLResourceManagerContainer::m_mutex;

GLResourceManagerContainer* GLResourceManagerContainer::m_instance = nullptr;

GLResourceManagerContainer* GLResourceManagerContainer::instance()
{
    if (nullptr == m_instance)
    {
        boost::unique_lock<boost::mutex> locker(m_mutex);
        if (nullptr == m_instance)
        {
            m_instance = new GLResourceManagerContainer();
        }
    }
    return m_instance;
}

GLResourceManagerContainer::~GLResourceManagerContainer()
{

}

GLProgramManagerPtr GLResourceManagerContainer::get_program_manager() const
{
    return m_pProgramMag;
}

GLBufferManagerPtr GLResourceManagerContainer::get_buffer_manager() const
{
    return m_pBufferMag;
}

void GLResourceManagerContainer::update_all()
{
    m_pProgramMag->update();
    m_pBufferMag->update();
}

GLResourceManagerContainer::GLResourceManagerContainer():
m_pProgramMag(new GLProgramManager()),
    m_pBufferMag(new GLBufferManager()),
    m_pTex1DMag(new GLTexture1DManager()),
    m_pTex2DMag(new GLTexture2DManager()),
    m_pTex3DMag(new GLTexture3DManager()),
    m_pVAOMag(new GLVAOManager()),
    m_pFBOMag(new GLFBOManager()),
    m_pTex1DArrayMag(new GLTexture1DArrayManager())
{

}

GLTexture1DManagerPtr GLResourceManagerContainer::get_texture_1d_manager() const
{
    return m_pTex1DMag;
}

GLTexture2DManagerPtr GLResourceManagerContainer::get_texture_2d_manager() const
{
    return m_pTex2DMag;
}

GLTexture3DManagerPtr GLResourceManagerContainer::get_texture_3d_manager() const
{
    return m_pTex3DMag;
}

GLVAOManagerPtr GLResourceManagerContainer::get_vao_manager() const
{
    return m_pVAOMag;
}

GLFBOManagerPtr GLResourceManagerContainer::get_fbo_manager() const
{
    return m_pFBOMag;
}

GLTexture1DArrayManagerPtr GLResourceManagerContainer::get_texture_1d_array_manager() const
{
    return m_pTex1DArrayMag;
}

MED_IMAGING_END_NAMESPACE