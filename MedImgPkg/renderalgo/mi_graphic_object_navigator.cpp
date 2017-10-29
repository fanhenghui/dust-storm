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

    OrthoCamera camera;
    if (_camera) {   
        Vector3 view =  _camera->get_view_direction();
        Vector3 up = _camera->get_up_direction();
        //ray casting result mapping flip vertically in scene
        view.z = -view.z;
        up.z = -up.z;
        camera.set_look_at(Point3::S_ZERO_POINT);
        const double dis = 100;
        Point3 eye = Point3::S_ZERO_POINT - view*dis;
        camera.set_eye(eye);
        camera.set_up_direction(up);
        camera.set_ortho(-1,1,-1,1,0,dis*2);
    }

    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(camera.get_view_projection_matrix()._m);

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glColor3f(1.0,1.0,0);
    float w = 0.6f;
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBegin(GL_QUADS);

    //patient coordinate
    //top 
    glVertex3f(-w, -w, w);
    glVertex3f(w, -w, w);
    glVertex3f(w, w, w);
    glVertex3f(-w, w, w);

    //bottom
    glVertex3f(-w, -w, -w);
    glVertex3f(-w, w, -w);
    glVertex3f(w, w, -w);
    glVertex3f(w, -w, -w);
    
    //left
    glVertex3f(-w, -w, -w);
    glVertex3f(-w, -w, w);
    glVertex3f(-w, w, w);
    glVertex3f(-w, w, -w);

    //right 
    glVertex3f(w, -w, -w);
    glVertex3f(w, w, -w);
    glVertex3f(w, w, w);
    glVertex3f(w, -w, w);

    //posterior
    glVertex3f(-w, w, -w);
    glVertex3f(-w, w, w);
    glVertex3f(w, w, w);
    glVertex3f(w, w, -w);

    //anterior
    glVertex3f(-w, -w, -w);
    glVertex3f(w, -w, -w);
    glVertex3f(w, -w, w);
    glVertex3f(-w, -w, w);
    
    glEnd();  

    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;
}

MED_IMG_END_NAMESPACE

