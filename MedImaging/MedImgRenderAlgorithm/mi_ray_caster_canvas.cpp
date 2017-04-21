#include "mi_ray_caster_canvas.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCasterCanvas::RayCasterCanvas():m_bInit(false),_width(32),_height(32)
{

}

RayCasterCanvas::~RayCasterCanvas()
{
    finialize();
}

void RayCasterCanvas::initialize()
{
    if (!m_bInit)
    {
        CHECK_GL_ERROR

        UIDType idFBO=0;
        m_pFBO = GLResourceManagerContainer::instance()->get_fbo_manager()->create_object(idFBO);
        m_pFBO->initialize();
        m_pFBO->set_target(GL_FRAMEBUFFER);

        UIDType idTexColor = 0;
        m_pColorAttach0 = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(idTexColor);
        m_pColorAttach0->initialize();
        m_pColorAttach0->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        m_pColorAttach0->load(GL_RGBA8 , _width , _height , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        UIDType idTexDepth = 0;
        m_pDepthAttach = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(idTexDepth);
        m_pDepthAttach->initialize();
        m_pDepthAttach->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        m_pDepthAttach->load(GL_DEPTH_COMPONENT16 , _width , _height , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

        //bind texture to FBO
        m_pFBO->bind();
        
        m_pFBO->attach_texture(GL_COLOR_ATTACHMENT0 , m_pColorAttach0);
        //m_pFBO->attach_texture(GL_COLOR_ATTACHMENT1 , m_pGrayAttach1);
        m_pFBO->attach_texture(GL_DEPTH_ATTACHMENT , m_pDepthAttach);

        m_pFBO->unbind();

        CHECK_GL_ERROR;

        //Create array
        m_pColorArray.reset(new RGBAUnit[_width*_height]);

        m_bInit = true;
    }
}

void RayCasterCanvas::finialize()
{
    if (m_bInit)
    {
        GLResourceManagerContainer::instance()->get_fbo_manager()->remove_object(m_pFBO->get_uid());
        GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(m_pColorAttach0->get_uid());
        GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(m_pDepthAttach->get_uid());

        GLResourceManagerContainer::instance()->get_fbo_manager()->update();
        GLResourceManagerContainer::instance()->get_texture_2d_manager()->update();
        m_bInit = false;
    }
}

void RayCasterCanvas::set_display_size( int iWidth , int iHeight )
{
    _width = iWidth;
    _height = iHeight;
}

GLFBOPtr RayCasterCanvas::get_fbo()
{
    return m_pFBO;
}

RGBAUnit* RayCasterCanvas::get_color_array()
{
    return m_pColorArray.get();
}

void RayCasterCanvas::update_fbo()
{
    if (m_bInit)
    {
        m_pColorAttach0->bind();
        m_pColorAttach0->load(GL_RGBA8 , _width , _height , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        m_pDepthAttach->bind();
        m_pDepthAttach->load(GL_DEPTH_COMPONENT16 , _width , _height , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

    }
}

void RayCasterCanvas::update_color_array()
{
    CHECK_GL_ERROR
    m_pColorAttach0->bind();
    m_pColorAttach0->update(0,0,_width , _height , GL_RGBA , GL_UNSIGNED_BYTE , m_pColorArray.get());
    CHECK_GL_ERROR
}

GLTexture2DPtr RayCasterCanvas::get_color_attach_texture()
{
    return m_pColorAttach0;
}

void RayCasterCanvas::debug_output_color(const std::string& sFileName)
{
    m_pColorAttach0->bind();
    std::unique_ptr<unsigned char[]> pRGBA(new unsigned char[_width*_height*4]);
    m_pColorAttach0->download(GL_RGBA , GL_UNSIGNED_BYTE , pRGBA.get());

    std::ofstream out(sFileName , std::ios::out | std::ios::binary);
    if (out.is_open())
    {
        out.write((char*)pRGBA.get(), _width*_height*4);
    }
    out.close();
}

void RayCasterCanvas::get_display_size(int& iWidth, int& iHeight) const
{
    iWidth = _width;
    iHeight = _height;
}

MED_IMAGING_END_NAMESPACE