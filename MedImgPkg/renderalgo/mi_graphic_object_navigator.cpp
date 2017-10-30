#include "mi_graphic_object_navigator.h"

#include "util/mi_file_util.h"
#include "arithmetic/mi_ortho_camera.h"
#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

GraphicObjectNavigator::GraphicObjectNavigator() : _has_init(false) {
    _width = 100;
    _height = 100;
    _x = 512 - _width - 20;
    _y = 512 - _height - 20;
}

GraphicObjectNavigator::~GraphicObjectNavigator() {

}

void GraphicObjectNavigator::initialize() {
    if(!_has_init) {
        const std::string navi_img_file("../config/navi_384_256_3.raw");
        const unsigned int img_size = 384*256*3;
        unsigned char* img_buffer = new unsigned char[img_size];
        if( 0 != FileUtil::read_raw(navi_img_file, img_buffer, img_size) ) {
            MI_RENDERALGO_LOG(MI_FATAL) << "load navigator image failed.";
        } else {
            UIDType uid = 0;
            _navi_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(uid);
            _navi_tex->set_description("navigator texture");
            _res_shield.add_shield<GLTexture2D>(_navi_tex);
            GLTextureCache::instance()->cache_load(GL_TEXTURE_2D, _navi_tex, GL_CLAMP_TO_BORDER, 
                GL_LINEAR, GL_RGB8, 384, 256, 1, GL_RGB, GL_UNSIGNED_BYTE, (char*)img_buffer);
        }
        _has_init = true;
    }
}

void GraphicObjectNavigator::set_navi_position(int x, int y, int width, int height) {
    _x = x;
    _y = y;
    _width = width;
    _height = height;
}

void GraphicObjectNavigator::render(int code) {
    if (_navi_tex == nullptr) {
        return;
    }

    OrthoCamera camera;
    if (_camera) {   
        Vector3 view = _camera->get_view_direction();
        Vector3 up = _camera->get_up_direction();
        camera.set_look_at(Point3::S_ZERO_POINT);
        const double dis = 1;
        Point3 eye = Point3::S_ZERO_POINT - view*dis;
        camera.set_eye(eye);
        camera.set_up_direction(up);
        camera.set_ortho(-1,1,-1,1,0,dis*2);
    }

    CHECK_GL_ERROR;
    glViewport(_x, _y, _width, _height);
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(camera.get_view_projection_matrix()._m);

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);

    const float w = 0.6f;
    glEnable(GL_TEXTURE_2D);
    _navi_tex->bind();
    glBegin(GL_QUADS);

    const float x_step = 0.33333f;
    const float y_step = 0.5f;
    //patient coordinate
    //head 
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(-w, -w, w);
    glTexCoord2f(x_step*3.0f, y_step);
    glVertex3f(w, -w, w);
    glTexCoord2f(x_step*3.0f, 0);
    glVertex3f(w, w, w);
    glTexCoord2f(x_step*2.0f, 0);
    glVertex3f(-w, w, w);

    //foot
    glTexCoord2f(x_step*2.0f, y_step*2.0f);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(-w, w, -w);
    glTexCoord2f(x_step*3.0f, y_step);
    glVertex3f(w, w, -w);
    glTexCoord2f(x_step*3.0f, y_step*2.0f);
    glVertex3f(w, -w, -w);
    
    //left
    glTexCoord2f(x_step, 0);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(-w, -w, w);
    glTexCoord2f(0, y_step);
    glVertex3f(-w, w, w);
    glTexCoord2f(0, 0);
    glVertex3f(-w, w, -w);

    //right
    glTexCoord2f(0, y_step);
    glVertex3f(w, -w, -w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(w, w, -w);
    glTexCoord2f(x_step, y_step*2);
    glVertex3f(w, w, w);
    glTexCoord2f(0, y_step*2);
    glVertex3f(w, -w, w);

    //posterior
    glTexCoord2f(x_step*2.0, 0);
    glVertex3f(-w, w, -w);
    glTexCoord2f(x_step*2.0, y_step);
    glVertex3f(-w, w, w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(w, w, w);
    glTexCoord2f(x_step, 0);
    glVertex3f(w, w, -w);

    //anterior
    glTexCoord2f(x_step, y_step);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step*2.0, y_step);
    glVertex3f(w, -w, -w);
    glTexCoord2f(x_step*2.0, y_step*2.0);
    glVertex3f(w, -w, w);
    glTexCoord2f(x_step, y_step*2.0);
    glVertex3f(-w, -w, w);
    
    glEnd();  

    _navi_tex->unbind();
    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;
}

MED_IMG_END_NAMESPACE

