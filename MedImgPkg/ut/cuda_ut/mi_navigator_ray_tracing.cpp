#include "GL/glew.h"

#include "log/mi_logger.h"

#include "util/mi_file_util.h"

#include "arithmetic/mi_ortho_camera.h"

#include "io/mi_configure.h"

#include "glresource/mi_gl_environment.h"
#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_resource_manager.h"
#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_time_query.h"
#include "glresource/mi_gl_utils.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#ifdef WIN32
#include "GL/glut.h"
#else
#include "GL/freeglut.h"
//#include "cuda_runtime.h"
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <cuda.h>  
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>

#include "mi_cuda_vr.h"

extern "C" void ray_tracing(Viewport& viewport, int width, int height, mat4& mat_viewmodel, mat4& mat_projection_inv, unsigned char* result);

using namespace medical_imaging;
namespace {
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;
    std::shared_ptr<GLTexture2D> _canvas_tex;

    unsigned char* _cuda_d_canvas = nullptr;
    unsigned char* _cuda_h_canvas = nullptr;

    int _width = 1024;
    int _height = 1024;
    int _button = -1;
    int _button_status = -1;
    Point2 _pre_pos;
}

void init() {
    Configure::instance()->set_processing_unit_type(GPU);
    GLUtils::set_check_gl_flag(true);

    //Global GL state
    GLUtils::set_check_gl_flag(true);
    GLUtils::set_pixel_pack_alignment(1);

    //Entry exit points
    _camera.reset(new OrthoCamera());
    _camera->set_ortho(-1, 1, -1, 1, -1, 1);
    _camera->set_eye(Point3(0, 0, -10));
    _camera->set_look_at(Point3(0, 0, 0));
    _camera->set_up_direction(Vector3(0, 1, 0));

    _camera_interactor.reset(new OrthoCameraInteractor(_camera));
    _camera_interactor->reset_camera();
    _camera_interactor->resize(_width, _height);

    //Canvas texture
    glEnable(GL_TEXTURE_2D);
    UIDType tex_uid = 0;
    _canvas_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(tex_uid);
    _canvas_tex->set_description("CUDA GL UT canvas texture.");
    _canvas_tex->initialize();
    _canvas_tex->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _canvas_tex->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    //CUDA canvas
    cudaMalloc(&_cuda_d_canvas, _width*_height * 4);
    _cuda_h_canvas = new unsigned char[_width*_height * 4];

}

static void Display() {
    try {
        CHECK_GL_ERROR;

        //CUDA process
        Matrix4 mat_view = _camera->get_view_matrix();
        Matrix4 mat_projection = _camera->get_projection_matrix();
        Matrix4 mat_projection_inv = mat_projection.get_inverse();
        mat4 mat4_v = matrix4_to_mat4(mat_view);
        mat4 mat4_pi = matrix4_to_mat4(mat_projection_inv);
        Viewport view_port(0, 0, _width, _height);

        {
            //test
            float w = 0.6;
            Point3 p000 = mat_view*Point3(-w, -w, -w);
            Point3 p001 = mat_view*Point3(-w, -w, w);
            Point3 p010 = mat_view*Point3(-w, w, -w);
            Point3 p011 = mat_view*Point3(-w, w, w);
            Point3 p100 = mat_view*Point3(w, -w, -w);
            Point3 p101 = mat_view*Point3(w, -w, w);
            Point3 p110 = mat_view*Point3(w, w, -w);
            Point3 p111 = mat_view*Point3(w, w, w);

            std::cout << "test done.";
        }

        {
            mat4 mat_viewmodel = matrix4_to_mat4(mat_view);
            float w = 0.6;
            float3 p000 = mat_viewmodel*make_float3(-w, -w, -w);
            float3 p001 = mat_viewmodel*make_float3(-w, -w, w);
            float3 p010 = mat_viewmodel*make_float3(-w, w, -w);
            float3 p011 = mat_viewmodel*make_float3(-w, w, w);
            float3 p100 = mat_viewmodel*make_float3(w, -w, -w);
            float3 p101 = mat_viewmodel*make_float3(w, -w, w);
            float3 p110 = mat_viewmodel*make_float3(w, w, -w);
            float3 p111 = mat_viewmodel*make_float3(w, w, w);

            std::cout << "test done.";

        }

        ray_tracing(view_port, _width, _height, mat4_v, mat4_pi, _cuda_d_canvas);
        cudaMemcpy(_cuda_h_canvas, _cuda_d_canvas, _width*_height * 4, cudaMemcpyDeviceToHost);

        //update texture
        glBindTexture(GL_TEXTURE_2D, _canvas_tex->get_id());
        _canvas_tex->update(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, _cuda_h_canvas, 0);

        glViewport(0, 0, _width, _height);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        
        glBegin(GL_QUADS);
        glTexCoord2d(0, 0);
        glVertex2d(-1.0, -1.0);
        glTexCoord2d(1, 0);
        glVertex2d(1.0, -1.0);
        glTexCoord2d(1, 1);
        glVertex2d(1.0, 1.0);
        glTexCoord2d(0, 1);
        glVertex2d(-1.0, 1.0);
        glEnd();

        glutSwapBuffers();
    }
    catch (Exception& e) {
        MI_LOG(MI_ERROR) << e.what();
        abort();
    }
}

static void Keyboard(unsigned char key, int x, int y) {
    switch (key) {
    default:
        break;
    }

    glutPostRedisplay();
}

static void Resize(int x, int y) {
    if (x == 0 || y == 0) {
        return;
    }

    _width = x;
    _height = y;
    glutPostRedisplay();
}

static void Idle() {
    glutPostRedisplay();
}

static void MouseClick(int button, int status, int x, int y) {
    _button = button;
    _button_status = status;
    _pre_pos = Point2(x, y);
    glutPostRedisplay();
}

static void MouseMotion(int x, int y) {
    Point2 pt(x, y);
    if (_button_status == GLUT_DOWN) {
        if (_button == GLUT_LEFT_BUTTON) {
            _camera_interactor->rotate(_pre_pos, pt, _width, _height);
        } else if (_button == GLUT_RIGHT_BUTTON) {
            _camera_interactor->zoom(_pre_pos, pt, _width, _height);
        }
    }

    _pre_pos = pt;
    glutPostRedisplay();
}

int mi_navigator_ray_tracing(int argc, char* argv[]) {
#ifndef WIN32
    chdir(dirname(argv[0]));
#endif

    try {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(_width, _height);

        glutCreateWindow("Navigator Ray Tracing");

        if (GLEW_OK != glewInit()) {
            MI_LOG(MI_FATAL) << "Init GLEW failed!\n";
            return -1;
        }

        GLEnvironment env;
        int major, minor;
        env.get_gl_version(major, minor);

        init();

        glutDisplayFunc(Display);
        glutReshapeFunc(Resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        glutMainLoop();

        return 0;
    }
    catch (const Exception& e) {
        MI_LOG(MI_FATAL) << e.what();
        return -1;
    }
}