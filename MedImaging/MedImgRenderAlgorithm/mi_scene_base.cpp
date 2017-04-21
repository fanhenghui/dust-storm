#include "mi_scene_base.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

SceneBase::SceneBase():_width(128),_height(128),m_bDirty(true),m_sName("Scene")
{
}

SceneBase::SceneBase(int width , int height):_width(width) , _height(height),m_bDirty(true)
{
}

SceneBase::~SceneBase()
{
    finalize();
}

void SceneBase::render_to_back()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pSceneFBO->get_id());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glDrawBuffer(GL_BACK);
    glBlitFramebuffer(0,0,_width, _height , 0,0,_width , _height , GL_COLOR_BUFFER_BIT , GL_NEAREST);
}

std::shared_ptr<CameraBase> SceneBase::get_camera()
{
    return _camera;
}

void SceneBase::initialize()
{
    if (!m_pSceneFBO)
    {
        //Init FBO
        CHECK_GL_ERROR;

        UIDType fbo_id=0;
        m_pSceneFBO = GLResourceManagerContainer::instance()->get_fbo_manager()->create_object(fbo_id);
        m_pSceneFBO->set_description("Scene base FBO");
        m_pSceneFBO->initialize();
        m_pSceneFBO->set_target(GL_FRAMEBUFFER);

        UIDType texture_color_id = 0;
        m_pSceneColorAttach0 = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(texture_color_id);
        m_pSceneColorAttach0->set_description("Scene base Color Attachment 0");
        m_pSceneColorAttach0->initialize();
        m_pSceneColorAttach0->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        m_pSceneColorAttach0->load(GL_RGBA8 , _width , _height , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        UIDType depth_color_id = 0;
        m_pSceneDepthAttach = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(depth_color_id);
        m_pSceneDepthAttach->set_description("Scene base Depth Attachment");
        m_pSceneDepthAttach->initialize();
        m_pSceneDepthAttach->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        m_pSceneDepthAttach->load(GL_DEPTH_COMPONENT16 , _width , _height , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

        //bind texture to FBO
        m_pSceneFBO->bind();

        m_pSceneFBO->attach_texture(GL_COLOR_ATTACHMENT0 , m_pSceneColorAttach0);
        m_pSceneFBO->attach_texture(GL_DEPTH_ATTACHMENT , m_pSceneDepthAttach);

        m_pSceneFBO->unbind();

        CHECK_GL_ERROR;
    }
}

void SceneBase::finalize()
{
    GLResourceManagerContainer::instance()->get_fbo_manager()->remove_object(m_pSceneFBO->get_uid());
    GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(m_pSceneColorAttach0->get_uid());
    GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(m_pSceneDepthAttach->get_uid());

    GLResourceManagerContainer::instance()->get_fbo_manager()->update();
    GLResourceManagerContainer::instance()->get_texture_2d_manager()->update();
}

void SceneBase::set_display_size(int width , int height)
{
    _width = width;
    _height = height;

    m_pSceneColorAttach0->bind();
    m_pSceneColorAttach0->load(GL_RGBA8 , _width , _height , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

    m_pSceneDepthAttach->bind();
    m_pSceneDepthAttach->load(GL_DEPTH_COMPONENT16 , _width , _height , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

    set_dirty(true);
}

void SceneBase::render(int)
{

}

void SceneBase::rotate(const Point2& pre_pt , const Point2& cur_pt)
{

}

void SceneBase::zoom(const Point2& pre_pt , const Point2& cur_pt)
{

}

void SceneBase::pan(const Point2& pre_pt , const Point2& cur_pt)
{

}

void SceneBase::get_display_size(int& width, int& height) const
{
    width = _width;
    height = _height;
}

void SceneBase::set_dirty(bool flag)
{
    m_bDirty = flag;
}

bool SceneBase::get_dirty() const
{
    return m_bDirty;
}

void SceneBase::set_name(const std::string& sName)
{
    m_sName = sName;
}

const std::string& SceneBase::get_name() const
{
    return m_sName;
}


MED_IMAGING_END_NAMESPACE