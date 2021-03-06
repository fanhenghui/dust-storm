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

#include "mi_cuda_graphic.h"
#include "mi_cuda_vr_common.h"

extern "C"
void ray_tracing_element_vertex_color(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, int ele_count, int* d_element, float4* d_color,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex);

extern "C"
void ray_tracing_element_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, int ele_count, int* d_element, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex);

extern "C"
void ray_tracing_triangle_vertex_color(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, float4* d_color,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex);

extern "C"
void ray_tracing_triangle_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex);

extern "C"
void ray_tracing_quad_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv, mat4 matmvp,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex, bool blend);


using namespace medical_imaging;
namespace {
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;
    std::shared_ptr<GLTexture2D> _canvas_tex;
    std::shared_ptr<GLTexture2D> _navigator_tex;
    cudaGLTextureWriteOnly _cuda_canvas_tex;
    cudaGLTextureReadOnly _cuda_navagator_tex;

    unsigned char* _cuda_d_canvas = nullptr;

    //graphic device memory(Global memory)
    float* _d_vertex = nullptr;
    float* _d_color = nullptr;
    float* _d_tex_coordinate = nullptr;
    int* _d_element = nullptr;

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
    _camera->set_ortho(-1, 1, -1, 1, 1, 5);
    _camera->set_eye(Point3(0, 0, 3));
    _camera->set_look_at(Point3(0, 0, 0));
    _camera->set_up_direction(Vector3(0, 1, 0));

    _camera_interactor.reset(new OrthoCameraInteractor(_camera));
    _camera_interactor->reset_camera();
    _camera_interactor->resize(_width, _height);

    //Canvas texture
    glEnable(GL_TEXTURE_2D);
    _canvas_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object("canvas");
    _canvas_tex->set_description("CUDA GL UT canvas texture.");
    _canvas_tex->initialize();
    _canvas_tex->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _canvas_tex->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    //CUDA canvas
    _cuda_canvas_tex.gl_tex_id = _canvas_tex->get_id();
    _cuda_canvas_tex.target = GL_TEXTURE_2D;
    register_image(_cuda_canvas_tex);
    cudaMalloc(&_cuda_d_canvas, _width*_height * 4);

    _canvas_tex->unbind();


    //texture
#ifdef WIN32
    const std::string navi_img_file("./config/resource/navi_384_256_3.raw");
#else
    const std::string navi_img_file("../config/resource/navi_384_256_3.raw");
#endif
    const unsigned int img_size = 384 * 256 * 3;
    unsigned char* img_buffer = new unsigned char[img_size];
    if (0 != FileUtil::read_raw(navi_img_file, img_buffer, img_size)) {
        MI_LOG(MI_FATAL) << "load navigator image failed.";
    }
    else {
        _navigator_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object("navigator");
        _navigator_tex->initialize();
        _navigator_tex->set_description("navigator texture");
        _navigator_tex->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        unsigned char* rgba4 = new unsigned char[384 * 256 * 4];
        for (int i = 0; i < 384 * 256; ++i) {
            rgba4[i * 4] = img_buffer[i * 3];
            rgba4[i * 4 + 1] = img_buffer[i * 3 + 1];
            rgba4[i * 4 + 2] = img_buffer[i * 3 + 2];
            rgba4[i * 4 + 3] = 255;
        }
        delete[] img_buffer;
        img_buffer = nullptr;
        _navigator_tex->load(GL_RGBA8, 384, 256, GL_RGBA, GL_UNSIGNED_BYTE, (char*)rgba4);
        delete[] rgba4;
        rgba4 = nullptr;

        _cuda_navagator_tex.target = GL_TEXTURE_2D;
        _cuda_navagator_tex.gl_tex_id = _navigator_tex->get_id();
        register_image(_cuda_navagator_tex);
        map_image(_cuda_navagator_tex);
        bind_texture(_cuda_navagator_tex, cudaReadModeNormalizedFloat, cudaFilterModeLinear, true);
        unmap_image(_cuda_navagator_tex);
        _navigator_tex->unbind();
    }
}

static void init_graphic() {
    const float w = 0.6f;
    const float x_step = 0.33333f;
    const float y_step = 0.5f;

    //---------------------------------//
    //element
    float ele_vertex[] = {
        -w, -w, -w,
        -w, -w, w,
        -w, w, -w,
        -w, w, w,
        w, -w, -w,
        w, -w, w,
        w, w, -w,
        w, w, w
    };

    float ele_color[] = {
        0, 0, 0, 1,//0
        0, 0, 1, 1,//1
        0, 1, 0, 1,//2
        0, 1, 1, 1,//3
        1, 0, 0, 1,//4
        1, 0, 1, 1,//5
        1, 1, 0, 1,//6
        1, 1, 1, 1//7
    };

    int element[] = {
        1,5,7,
        1,7,3,
        0,2,6,
        0,6,4,
        0,1,3,
        0,3,2,
        4,6,7,
        4,7,5,
        2,3,7,
        2,7,6,
        0,4,5,
        0,5,1
    };

    //---------------------------------//
    //triangle
    float vertex[] = {
        -w, -w, w,
        w, -w, w,
        w, w, w,
        -w, w, w,
        -w, -w, -w,
        -w, w, -w,
        w, w, -w,
        w, -w, -w,
        -w, -w, -w,
        -w, -w, w,
        -w, w, w,
        -w, w, -w,
        w, -w, -w,
        w, w, -w,
        w, w, w,
        w, -w, w,
        -w, w, -w,
        -w, w, w,
        w, w, w,
        w, w, -w,
        -w, -w, -w,
        w, -w, -w,
        w, -w, w,
        -w, -w, w
    };

    float tex_coordinate[] = {
        x_step*2.0f, y_step,
        x_step*3.0f, y_step,
        x_step*3.0f, 0,
        x_step*2.0f, 0,
        x_step*2.0f, y_step*2.0f,
        x_step*2.0f, y_step,
        x_step*3.0f, y_step,
        x_step*3.0f, y_step*2.0f,
        x_step, 0,
        x_step, y_step,
        0, y_step,
        0, 0,
        0, y_step,
        x_step, y_step,
        x_step, y_step * 2,
        0, y_step * 2,
        x_step*2.0f, 0,
        x_step*2.0f, y_step,
        x_step, y_step,
        x_step, 0,
        x_step, y_step,
        x_step*2.0f, y_step,
        x_step*2.0f, y_step*2.0f,
        x_step, y_step*2.0f
    };

    

    cudaMalloc(&_d_vertex, sizeof(vertex));
    cudaMemcpy(_d_vertex, vertex, sizeof(vertex), cudaMemcpyHostToDevice);

    //cudaMalloc(&_d_color, sizeof(color));
    //cudaMemcpy(_d_color, color, sizeof(color), cudaMemcpyHostToDevice);

    //cudaMalloc(&_d_element, sizeof(element));
    //cudaMemcpy(_d_element, element, sizeof(element), cudaMemcpyHostToDevice);

    cudaMalloc(&_d_tex_coordinate, sizeof(tex_coordinate));
    cudaMemcpy(_d_tex_coordinate, tex_coordinate, sizeof(tex_coordinate), cudaMemcpyHostToDevice);
  
}

static void Display2() {
    glViewport(0, 0, _width, _height);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(_camera->get_view_projection_matrix()._m);
//    glLoadIdentity();

    CHECK_GL_ERROR;

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);

    CHECK_GL_ERROR;

    const float w = 0.6f;
    const float x_step = 0.33333f;
    const float y_step = 0.5f;

    _navigator_tex->bind();
    glBegin(GL_QUADS);

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
    glTexCoord2f(x_step, y_step * 2);
    glVertex3f(w, w, w);
    glTexCoord2f(0, y_step * 2);
    glVertex3f(w, -w, w);

    //posterior
    glTexCoord2f(x_step*2.0f, 0);
    glVertex3f(-w, w, -w);
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(-w, w, w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(w, w, w);
    glTexCoord2f(x_step, 0);
    glVertex3f(w, w, -w);

    //anterior
    glTexCoord2f(x_step, y_step);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(w, -w, -w);
    glTexCoord2f(x_step*2.0f, y_step*2.0f);
    glVertex3f(w, -w, w);
    glTexCoord2f(x_step, y_step*2.0f);
    glVertex3f(-w, -w, w);

    glEnd();
    CHECK_GL_ERROR;

    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;

    glutSwapBuffers();
}

static void Display() {
    try {
        CHECK_GL_ERROR;

        //CUDA process
        Matrix4 mat_view = _camera->get_view_matrix();
        Matrix4 mat_projection = _camera->get_projection_matrix();
        Matrix4 mat_projection_inv = mat_projection.get_inverse();
        Matrix4 mat_mvp = mat_projection*mat_view;
        mat4 mat4_v = matrix4_to_mat4(mat_view);
        mat4 mat4_pi = matrix4_to_mat4(mat_projection_inv);
        mat4 mat4_mvp = matrix4_to_mat4(mat_mvp);
        //Viewport view_port(_width/3*2, _height/3*2, _width/3, _height/3);
        Viewport view_port(0, 0, _width, _height);

        //debug
       /* {

            const float w = 0.6f;
            Point3 vertexs[] = {
                Point3(-w, -w, -w),
                Point3(-w, -w, w),
                Point3(-w, w, -w),
                Point3(-w, w, w),
                Point3(w, -w, -w),
                Point3(w, -w, w),
                Point3(w, w, -w),
                Point3(w, w, w)
            };
            std::cout << "{\n";
            for (int i = 0; i < 8; ++i) {
                Point3 tmp = mat_mvp.transform(vertexs[i]);
                std::cout << " pt: " << tmp << std::endl;
            }
            std::cout << std::endl;
            for (int i = 0; i < 8; ++i) {
                float3 tmp = mat4_mvp*make_float3(vertexs[i].x, vertexs[i].y, vertexs[i].z);
                std::cout << " pt: " << tmp.x << " , " << tmp.y << " , " << tmp.z  << std::endl;
            }
            std::cout << "}\n";
        }*/

        //ray_tracing(view_port, _width, _height, mat4_v, mat4_pi, _cuda_d_canvas, _cuda_canvas_tex);
        //ray_tracing_vertex_color(view_port, _width, _height, mat4_v, mat4_pi, 8, (float3*)_d_vertex, 36, _d_element, (float4*)_d_color, _cuda_d_canvas, _cuda_canvas_tex);

        map_image(_cuda_navagator_tex);
        ray_tracing_quad_vertex_mapping(view_port, _width, _height, mat4_v, mat4_pi, mat4_mvp, 24, (float3*)_d_vertex, (float2*)_d_tex_coordinate, _cuda_navagator_tex, _cuda_d_canvas, _cuda_canvas_tex, false);
        unmap_image(_cuda_navagator_tex);

        //update texture
        glViewport(0, 0, _width, _height);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, _canvas_tex->get_id());
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

static void Finalize() {
    unmap_image(_cuda_canvas_tex);
    cudaFree(_cuda_d_canvas);
    _cuda_d_canvas = NULL;
}

int mi_simple_ray_tracing(int argc, char* argv[]) {
#ifndef WIN32
    chdir(dirname(argv[0]));
#endif

    try {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(_width, _height);

        glutCreateWindow("Simple Ray Tracing");

        if (GLEW_OK != glewInit()) {
            MI_LOG(MI_FATAL) << "Init GLEW failed!\n";
            return -1;
        }

        GLEnvironment env;
        int major, minor;
        env.get_gl_version(major, minor);

        init();
        init_graphic();

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