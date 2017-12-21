#include "GL/glew.h"

#include "io/mi_configure.h"
#include "util/mi_file_util.h"
#include "log/mi_logger.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_run_length_operator.h"

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
#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_mpr_entry_exit_points.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_ray_caster_canvas.h"
#include "renderalgo/mi_transfer_function_loader.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_brick_pool.h"
#include "renderalgo/mi_render_algo_logger.h"
#include "renderalgo/mi_vr_entry_exit_points.h"

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


#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"

#include "mi_cuda_vr.h"

extern "C" int bind_gl_texture(cudaGLTexture& tex);
extern "C" int ray_cast(cudaGLTexture entry_tex, cudaGLTexture exit_tex, cudaVolumeInfos volume_info, unsigned char* d_result, unsigned char* h_result);

using namespace medical_imaging;
namespace {
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::shared_ptr<VREntryExitPoints> _entry_exit_points;
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;
    std::shared_ptr<GLTexture2D> _canvas_tex;

    //CUDA resource
    cudaVolumeInfos _cuda_volume_infos;
    cudaGLTexture _cuda_entry_points;
    cudaGLTexture _cuda_exit_points;
    unsigned char* _cuda_d_canvas = nullptr;
    unsigned char* _cuda_h_canvas = nullptr;

    float _ww = 1500.0f;
    float _wl = -400.0f;

    int _width = 1024;
    int _height = 1024;
    int _button = -1;
    int _button_status = -1;
    Point2 _pre_pos;

    std::shared_ptr<GLTexture2D> _tex_entry_points;


#ifdef WIN32
    const std::string root = "E:/data/MyData/demo/lung/";
#else
    const std::string root = "/home/wangrui22/data/demo/lung/";
#endif

    std::vector<std::string> GetFiles() {
        std::vector<std::string> files;
        std::set<std::string> dcm_postfix;
        dcm_postfix.insert(".dcm");
        FileUtil::get_all_file_recursion(root+"/LIDC-IDRI-1002", dcm_postfix, files);
        return files;
    }

    void Finalize() {
        _data_header.reset();
        _volume_data.reset();
        _volume_infos.reset();
    }

    void Init() {
        Configure::instance()->set_processing_unit_type(GPU);
        GLUtils::set_check_gl_flag(true);
#ifdef WIN32
        Logger::instance()->bind_config_file("./config/log_config");
#else
        Logger::instance()->bind_config_file("../config/log_config");
#endif

        Logger::instance()->initialize();

        std::vector<std::string> files = GetFiles();
        DICOMLoader loader;
        loader.load_series(files, _volume_data, _data_header);
        const unsigned int data_len = _volume_data->_dim[0]*_volume_data->_dim[1]*_volume_data->_dim[2];
        _volume_infos.reset(new VolumeInfos());
        _volume_infos->set_data_header(_data_header);
        _volume_infos->set_volume(_volume_data);
        _volume_infos->refresh();

        _cuda_volume_infos.dim.x = _volume_data->_dim[0];
        _cuda_volume_infos.dim.y = _volume_data->_dim[1];
        _cuda_volume_infos.dim.z = _volume_data->_dim[2];
    }

    void InitGL() {
        //Global GL state
        GLUtils::set_check_gl_flag(true);
        GLUtils::set_pixel_pack_alignment(1);

        //Entry exit points
        _camera.reset(new OrthoCamera());
        std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
        camera_cal->init_vr_placement(_camera);
        _camera_interactor.reset(new OrthoCameraInteractor(_camera));
        _camera_interactor->reset_camera();
        _camera_interactor->resize(_width, _height);

        _entry_exit_points.reset(new VREntryExitPoints());
        _entry_exit_points->set_display_size(_width,_height);
        _entry_exit_points->set_strategy(GPU_BASE);
        _entry_exit_points->initialize();
        _entry_exit_points->set_proxy_geometry(PG_BRICKS);
        _entry_exit_points->set_camera(_camera);
        _entry_exit_points->set_camera_calculator(camera_cal);

        std::shared_ptr<ImageData> volume = _volume_infos->get_volume();
        _entry_exit_points->set_volume_data(volume);
        _entry_exit_points->set_brick_pool(_volume_infos->get_brick_pool());
        AABB default_aabb;
        default_aabb._min = Point3::S_ZERO_POINT;
        default_aabb._max.x = static_cast<double>(volume->_dim[0]);
        default_aabb._max.y = static_cast<double>(volume->_dim[1]);
        default_aabb._max.z = static_cast<double>(volume->_dim[2]);
        _entry_exit_points->set_bounding_box(default_aabb);
        _entry_exit_points->set_brick_pool(_volume_infos->get_brick_pool());

        _entry_exit_points->set_brick_filter_item(BF_WL);
        _entry_exit_points->set_window_level(_ww, _wl, 0, true);

        _tex_entry_points = _entry_exit_points->get_entry_points_texture();


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

        
        //generate cuda texture
        _entry_exit_points->get_entry_points_texture()->bind();
        _cuda_entry_points.gl_tex_id = _entry_exit_points->get_entry_points_texture()->get_id();
        _cuda_entry_points.target = GL_TEXTURE_2D;
        _cuda_entry_points.width = _width;
        _cuda_entry_points.height = _height;
        if(0 != bind_gl_texture(_cuda_entry_points)) {
            MI_LOG(MI_ERROR) << "[CUDA] " << "bind GL texture failed."; 
        }

        _entry_exit_points->get_exit_points_texture()->bind();
        _cuda_exit_points.gl_tex_id = _entry_exit_points->get_exit_points_texture()->get_id();
        _cuda_exit_points.target = GL_TEXTURE_2D;
        _cuda_exit_points.width = _width;
        _cuda_exit_points.height = _height;
        if(0 != bind_gl_texture(_cuda_exit_points)) {
            MI_LOG(MI_ERROR) << "[CUDA] " << "bind GL texture failed.";
        }

        _cuda_h_canvas = new unsigned char[_width*_height*4];
        if(cudaSuccess != cudaMalloc(&_cuda_d_canvas, _width*_height*4) ) {
            MI_LOG(MI_ERROR) << "[CUDA] " << "malloc canvas device memory failed.";
        }
        
    }

    void Display() {
        try {
            CHECK_GL_ERROR;

            //calcualte entry exit pionts
            _entry_exit_points->calculate_entry_exit_points();

            //CUDA process
            ray_cast(_cuda_entry_points, _cuda_entry_points, _cuda_volume_infos, _cuda_d_canvas, _cuda_h_canvas);
            CHECK_GL_ERROR;

            //update texture
            _canvas_tex->update(0,0,_width,_height, GL_RGBA, GL_UNSIGNED_BYTE, _cuda_h_canvas,0);
            
            glViewport(0, 0, _width, _height);
            glClearColor(0.0, 0.0, 0.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT);

            glBindTexture(GL_TEXTURE_2D, _canvas_tex->get_id());
            //glBindTexture(GL_TEXTURE_2D, _entry_exit_points->get_entry_points_texture()->get_id());
            glBegin(GL_QUADS);
            glTexCoord2d(0,0);
            glVertex2d(-1.0,-1.0);
            glTexCoord2d(1,0);
            glVertex2d(1.0,-1.0);
            glTexCoord2d(1,1);
            glVertex2d(1.0,1.0);
            glTexCoord2d(0,1);
            glVertex2d(-1.0,1.0);
            glEnd();

            glutSwapBuffers();
        } catch (Exception& e) {
            MI_RENDERALGO_LOG(MI_ERROR) << e.what();
            abort();
        }
    }

    void Keyboard(unsigned char key, int x, int y) {
        switch (key) {
        default:
            break;
        }

        glutPostRedisplay();
    }

    void resize(int x, int y) {
        if (x == 0 || y == 0) {
            return;
        }

        _width = x;
        _height = y;
        glutPostRedisplay();
    }

    void Idle() {
        glutPostRedisplay();
    }

    void MouseClick(int button, int status, int x, int y) {
        _button = button;
        _button_status = status;
        _pre_pos = Point2(x,y);
        glutPostRedisplay();
    }

    void MouseMotion(int x, int y) {
        Point2 pt(x,y);
        if (_button_status == GLUT_DOWN) {
            if (_button == GLUT_LEFT_BUTTON) {
                _camera_interactor->rotate(_pre_pos, pt, _width, _height);
            } else if (_button == GLUT_RIGHT_BUTTON) {
                _camera_interactor->zoom(_pre_pos, pt, _width, _height);
            } else if (_button == GLUT_MIDDLE_BUTTON) {
                _ww += (float)(x - _pre_pos.x);
                _wl += (float)(_pre_pos.y - y);
                _ww = _ww < 1.0f ? 1.0f : _ww;
                _entry_exit_points->set_window_level(_ww, _wl, 0, true);
            }
        }
        
        _pre_pos = pt;
        glutPostRedisplay();
    }
}

int mi_cuda_vr(int argc, char* argv[]) {
#ifndef WIN32
    chdir(dirname(argv[0]));
#endif

    try {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(_width, _height);

        glutCreateWindow("CUDA VR");

        if (GLEW_OK != glewInit()) {
            MI_RENDERALGO_LOG(MI_FATAL) << "Init GLEW failed!\n";
            return -1;
        }

        GLEnvironment env;
        int major, minor;
        env.get_gl_version(major, minor);

        Init();
        InitGL();

        glutDisplayFunc(Display);
        glutReshapeFunc(resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        glutMainLoop();

        Finalize();

        return 0;
    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << e.what();
        return -1;
    }
}