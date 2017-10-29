#include "mi_graphic_object_navigator.h"
#include "glresource/mi_gl_utils.h"
#include "arithmetic/mi_ortho_camera.h"

MED_IMG_BEGIN_NAMESPACE

GraphicObjectNavigator::GraphicObjectNavigator() {
    _s_width = 512;
    _s_height = 512;
    _width = _s_width/5;
    _height = _s_height/5;
    _x = _s_width - _width;
    _y = _s_height - _height;
}

GraphicObjectNavigator::~GraphicObjectNavigator() {

}

void GraphicObjectNavigator::initialize() {
    //TODO load tex mapping    
}

void GraphicObjectNavigator::set_scene_display_size(int width, int height) {
    _s_width = width;
    _s_height = height;
}

void GraphicObjectNavigator::set_navi_position(int x, int y, int width, int height) {
    _x = x;
    _y = y;
    _width = width;
    _height = height;
}

void GraphicObjectNavigator::render(int code) {
    CHECK_GL_ERROR;

    glViewport(_x, _y, _width, _height);

    // OrthoCamera camera;
    // if (_camera) {   
    //     Vector3 view =  _camera->get_view_direction();
    // }

    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(camera.get_view_projection_matrix()._m);

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glColor3f(1.0,1.0,0);
    float w = 0.8f;
    glBegin(GL_QUADS);
    glVertex3f(-w, -w ,0);
    glVertex3f(w, -w ,0);
    glVertex3f(w, w ,0);
    glVertex3f(-w, w ,0);
    glEnd();  

    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;
}

MED_IMG_END_NAMESPACE

