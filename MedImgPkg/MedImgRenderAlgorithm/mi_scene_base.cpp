#include "mi_scene_base.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE

SceneBase::SceneBase():_width(128),_height(128),_dirty(true),_name("Scene"),_front_buffer_id(0)
{
    _image_buffer[0].reset(new char[_width*_height*4]);
    _image_buffer[1].reset(new char[_width*_height*4]);
}

SceneBase::SceneBase(int width , int height):_width(width) , _height(height),_dirty(true)
{
}

SceneBase::~SceneBase()
{
    finalize();
}

void SceneBase::render_to_back()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER , _scene_fbo->get_id());
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
    if (!_scene_fbo)
    {
        //Init FBO
        CHECK_GL_ERROR;

        UIDType fbo_id=0;
        _scene_fbo = GLResourceManagerContainer::instance()->get_fbo_manager()->create_object(fbo_id);
        _scene_fbo->set_description("Scene base FBO");
        _scene_fbo->initialize();
        _scene_fbo->set_target(GL_FRAMEBUFFER);

        UIDType texture_color_id = 0;
        _scene_color_attach_0 = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(texture_color_id);
        _scene_color_attach_0->set_description("Scene base Color Attachment 0");
        _scene_color_attach_0->initialize();
        _scene_color_attach_0->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        _scene_color_attach_0->load(GL_RGBA8 , _width , _height , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

        UIDType depth_color_id = 0;
        _scene_depth_attach = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(depth_color_id);
        _scene_depth_attach->set_description("Scene base Depth Attachment");
        _scene_depth_attach->initialize();
        _scene_depth_attach->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        _scene_depth_attach->load(GL_DEPTH_COMPONENT16 , _width , _height , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

        //bind texture to FBO
        _scene_fbo->bind();

        _scene_fbo->attach_texture(GL_COLOR_ATTACHMENT0 , _scene_color_attach_0);
        _scene_fbo->attach_texture(GL_DEPTH_ATTACHMENT , _scene_depth_attach);

        _scene_fbo->unbind();

        CHECK_GL_ERROR;
    }
}

void SceneBase::finalize()
{
    GLResourceManagerContainer::instance()->get_fbo_manager()->remove_object(_scene_fbo->get_uid());
    GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(_scene_color_attach_0->get_uid());
    GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(_scene_depth_attach->get_uid());

    GLResourceManagerContainer::instance()->get_fbo_manager()->update();
    GLResourceManagerContainer::instance()->get_texture_2d_manager()->update();
}

void SceneBase::set_display_size(int width , int height)
{
    _width = width;
    _height = height;

    _image_buffer[0].reset(new char[_width*_height*4]);
    _image_buffer[1].reset(new char[_width*_height*4]);

    _scene_color_attach_0->bind();
    _scene_color_attach_0->load(GL_RGBA8 , _width , _height , GL_RGBA , GL_UNSIGNED_BYTE , nullptr);

    _scene_depth_attach->bind();
    _scene_depth_attach->load(GL_DEPTH_COMPONENT16 , _width , _height , GL_DEPTH_COMPONENT , GL_UNSIGNED_SHORT , nullptr);

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
    _dirty = flag;
}

bool SceneBase::get_dirty() const
{
    return _dirty;
}

void SceneBase::set_name(const std::string& name)
{
    _name = name;
}

const std::string& SceneBase::get_name() const
{
    return _name;
}

void SceneBase::download_image_buffer()
{
    //download FBO to back buffer
    boost::mutex::scoped_lock locker(_write_mutex);
    _scene_color_attach_0->download(GL_RGBA , GL_UNSIGNED_BYTE , _image_buffer[1 - _front_buffer_id].get() , 0 );
}

void SceneBase::swap_image_buffer()
{
    boost::mutex::scoped_lock locker0(_read_mutex);
    boost::mutex::scoped_lock locker1(_write_mutex);
    _front_buffer_id = 1 - _front_buffer_id;
}

void SceneBase::get_image_buffer(void* buffer)
{
    //Get front buffer
    boost::mutex::scoped_lock locker(_read_mutex);
    buffer = _image_buffer[_front_buffer_id].get();
}


MED_IMG_END_NAMESPACE