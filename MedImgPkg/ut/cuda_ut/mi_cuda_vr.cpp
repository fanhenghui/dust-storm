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

#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"

using namespace medical_imaging;
namespace {
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<VolumeInfos> _volumeinfos;

    int _ww;
    int _wl;

    int _width = 1024;
    int _height = 1024;
    int _iButton = -1;
    Point2 _ptPre;
    int _iTestCode = 0;
    bool _pan_status = false;

    int _act_label_idx = 0;
    std::vector<unsigned char> _vis_labels;

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
        _volumeinfos.reset();
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
        _volumeinfos.reset(new VolumeInfos());
        _volumeinfos->set_data_header(_data_header);
        _volumeinfos->set_volume(_volume_data);
    }

    void Display() {
        try {
            CHECK_GL_ERROR;

            GLUtils::set_pixel_pack_alignment(1);

            CHECK_GL_ERROR;

            glViewport(0, 0, _width, _height);
            glClearColor(0.0, 0.0, 0.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT);

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
        //glutPostRedisplay();
    }

    void MouseClick(int button, int status, int x, int y) {
        glutPostRedisplay();
    }

    void MouseMotion(int x, int y) {
        glutPostRedisplay();
    }
}

int mi_cuda_vr(int argc, char* argv[]) {
#ifndef WIN32
    chdir(dirname(argv[0]));
#endif

    try {

        Init();

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