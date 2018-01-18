#include "GL/glew.h"

#include "log/mi_logger.h"

#include "util/mi_file_util.h"

#include "io/mi_configure.h"
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

#include "cudaresource/mi_cuda_time_query.h"
#include "cudaresource/mi_cuda_resource_manager.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_mpr_entry_exit_points.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_ray_caster_canvas.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_fps.h"
#include "renderalgo/mi_render_algo_logger.h"

#ifdef WIN32
#include "GL/glut.h"
#else
#include "GL/freeglut.h"
//#include "cuda_runtime.h"
#endif

#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"

using namespace medical_imaging;
namespace {
std::shared_ptr<ImageDataHeader> _data_header;
std::shared_ptr<ImageData> _volume_data;
std::shared_ptr<VolumeInfos> _volumeinfos;

std::shared_ptr<MPRScene> _scene;

std::shared_ptr<GLTimeQuery> _time_query;
std::shared_ptr<CudaTimeQuery> _time_query_2;

int _width = 1033;
int _height = 616;
int _iButton = -1;
Point2 _pt_pre;
int _iTestCode = 0;
GPUPlatform _gpu_platform = CUDA_BASE;

int _sum_page = 0;
int _cur_page = 0;

bool _render_to_back = true;

FPS _fps;

#ifdef WIN32
const std::string root = "E:/data/MyData/demo/lung/";
#else
const std::string root = "/home/wangrui22/data/demo/lung/";
#endif

std::vector<std::string> GetFiles() {
    std::vector<std::string> files;
    std::set<std::string> dcm_postfix;
    dcm_postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(root, dcm_postfix, files);
    return files;
}

void Init() {
    Configure::instance()->set_processing_unit_type(GPU);
    GLUtils::set_check_gl_flag(true);

    GLUtils::set_pixel_pack_alignment(1);
    GLUtils::set_pixel_unpack_alignment(1);

#ifdef WIN32
    Logger::instance()->bind_config_file("./config/log_config");
#else
    Logger::instance()->bind_config_file("../config/log_config");
#endif
    Logger::instance()->initialize();

    std::vector<std::string> files = GetFiles();
    DICOMLoader loader;
    loader.load_series(files, _volume_data, _data_header);
    const unsigned int data_len = _volume_data->_dim[0] * _volume_data->_dim[1] * _volume_data->_dim[2];
    _sum_page = (int)_volume_data->_dim[2];

    _volumeinfos.reset(new VolumeInfos(GPU_BASE, _gpu_platform));
    _volumeinfos->set_data_header(_data_header);
    _volumeinfos->set_volume(_volume_data);

    // Create empty mask
    std::shared_ptr<ImageData> mask_data(new ImageData());
    _volume_data->shallow_copy(mask_data.get());
    mask_data->_channel_num = 1;
    mask_data->_data_type = medical_imaging::UCHAR;
    mask_data->mem_allocate();
    char* mask_raw = (char*)mask_data->get_pixel_pointer();
    std::ifstream in(root + "/mask.raw", std::ios::in);
    if (in.is_open()) {
        in.read(mask_raw, data_len);
        in.close();
    }
    else {
        memset(mask_raw, 1, data_len);
    }

    std::set<unsigned char> target_label_set;
    std::ifstream in2(root + "/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.rle", std::ios::binary | std::ios::in);
    if (in2.is_open()) {
        in2.seekg(0, in2.end);
        const int code_len = (int)in2.tellg();
        in2.seekg(0, in2.beg);
        unsigned int *code_buffer = new unsigned int[code_len];
        in2.read((char*)code_buffer, code_len);
        in2.close();
        unsigned char* mask_target = new unsigned char[data_len];

        RunLengthOperator run_length_op;
        if (0 == run_length_op.decode(code_buffer, code_len / sizeof(unsigned int), mask_target, data_len)) {
            FileUtil::write_raw(root + "./nodule.raw", mask_target, data_len);
            printf("load target mask done.\n");
            for (unsigned int i = 0; i < data_len; ++i) {
                if (mask_target[i] != 0) {
                    mask_raw[i] = mask_target[i] + 1;
                    target_label_set.insert(mask_target[i] + 1);
                }
            }
        }
        delete[] mask_target;
    }

    FileUtil::write_raw(root + "/target_mask.raw", mask_raw, data_len);

    _volumeinfos->set_mask(mask_data);

    _scene.reset(new MPRScene(_width, _height, GPU_BASE, _gpu_platform));
    const float PRESET_CT_LUNGS_WW = 1500;
    const float PRESET_CT_LUNGS_WL = -400;

    _scene->set_volume_infos(_volumeinfos);
    _scene->set_sample_step(1.0);
    _scene->set_global_window_level(PRESET_CT_LUNGS_WW, PRESET_CT_LUNGS_WL);
    _scene->set_composite_mode(COMPOSITE_AVERAGE);
    _scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
    _scene->set_mask_mode(MASK_NONE);
    _scene->set_interpolation_mode(LINEAR);
    _scene->place_mpr(TRANSVERSE);
    _cur_page = _sum_page/2;
    _scene->page_to(_cur_page);

    // Time query
    _time_query = GLResourceManagerContainer::instance()
                  ->get_time_query_manager()
                  ->create_object("TQ 1");
    _time_query->initialize();

    _time_query_2 = CudaResourceManager::instance()->create_cuda_time_query("TQ");
    _time_query_2->initialize();

    //Mask overlay
    if (!target_label_set.empty()) {
        std::vector<unsigned char> visible_labels;
        for (auto it = target_label_set.begin(); it != target_label_set.end(); ++it) {
            unsigned char label = *it;
            _scene->set_mask_overlay_color(RGBAUnit(255, 0, 0, 255), label);
            visible_labels.push_back(label);
        }
        _scene->set_mask_overlay_opacity(0.75f);
        _scene->set_mask_overlay_mode(MASK_OVERLAY_ENABLE);
        _scene->set_visible_labels(visible_labels);
    }

    MI_RENDERALGO_LOG(MI_DEBUG) << "TQ 3";
}

void Display() {
    try {

        GLUtils::set_pixel_pack_alignment(1);

        glViewport(0, 0, _width, _height);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        //_time_query->begin();        
        _time_query_2->begin();

        _scene->set_test_code(_iTestCode);
        _scene->set_dirty(true);

        _scene->render();

        if (_render_to_back) {
            _scene->render_to_back();
        }
        

        // glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);

        _scene->download_image_buffer();
        _scene->swap_image_buffer();
        unsigned char* buffer = nullptr;
        int buffer_size = 0;
        _scene->get_image_buffer(buffer, buffer_size);

#ifdef WIN32
        FileUtil::write_raw("D:/temp/output_ut.jpeg", buffer, buffer_size);
#else
        // FileUtil::write_raw("/home/wr/data/output_ut.jpeg",buffer , buffer_size);
#endif
        // std::cout << "rendering time : " << _time_query->end() << std::endl;
        const float render_time = _time_query_2->end();

        int fps = _fps.fps(render_time);
        static int tiker = 0;
        tiker++;
        if (tiker > 50) {
            MI_RENDERALGO_LOG(MI_INFO) << "FPS:  " << fps;
            tiker = 0;
        }

        glutSwapBuffers();
    } catch (Exception& e) {
        std::cout << e.what();
        abort();
    }
}

void Keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 't': {
        // std::cout << "W H :" << _width << " " << _height << std::endl;
        // m_pMPREE->debug_output_entry_points("D:/entry_exit.rgb.raw");
        std::shared_ptr<OrthoCamera> camera =
            std::dynamic_pointer_cast<OrthoCamera>(_scene->get_camera());
        int cur_page =
            _scene->get_camera_calculator()->get_orthogonal_mpr_page(camera);
        std::cout << "current page : " << cur_page << std::endl;

        if (cur_page >= (int)_volume_data->_dim[2] - 2) {
            _scene->page_to(1);
        } else {
            _scene->page(1);
        }

        break;
    }
    case 'b': {
        _render_to_back = !_render_to_back;
        break;
    }

    // case 'a':
    //     {
    //         m_pCameraCal->init_mpr_placement(_camera , TRANSVERSE ,
    //         Point3(0,0,0));
    //         m_pCameraInteractor->set_initial_status(_camera);
    //         m_pCameraInteractor->resize(_width , _height);
    //         break;
    //     }
    // case 's':
    //     {
    //         m_pCameraCal->init_mpr_placement(_camera , SAGITTAL ,
    //         Point3(0,0,0));
    //         m_pCameraInteractor->set_initial_status(_camera);
    //         m_pCameraInteractor->resize(_width , _height);
    //         break;
    //     }
    // case 'c':
    //     {
    //         m_pCameraCal->init_mpr_placement(_camera , CORONAL, Point3(0,0,0));
    //         m_pCameraInteractor->set_initial_status(_camera);
    //         m_pCameraInteractor->resize(_width , _height);
    //         break;
    //     }
    case 'f': {
        _iTestCode = 1 - _iTestCode;
        break;
    }

    default:
        break;
    }

    glutPostRedisplay();
}

void Resize(int x, int y) {
    if (x == 0 || y == 0) {
        return;
    }

    _width = x;
    _height = y;
    _scene->set_display_size(_width, _height);
    glutPostRedisplay();
}

void Idle() {
    glutPostRedisplay();
}

void MouseClick(int button, int status, int x, int y) {
    x = x < 0 ? 0 : x;
    x = x > _width - 1 ? _width - 1 : x;
    y = y < 0 ? 0 : y;
    y = y > _height - 1 ? _height - 1 : y;

    _iButton = button;

    if (_iButton == GLUT_LEFT_BUTTON) {

    } else if (_iButton == GLUT_MIDDLE_BUTTON) {

    } else if (_iButton == GLUT_RIGHT_BUTTON) {
    }

    _pt_pre = Point2(x, y);
    glutPostRedisplay();
}

void MouseMotion(int x, int y) {
    x = x < 0 ? 0 : x;
    x = x > _width - 1 ? _width - 1 : x;
    y = y < 0 ? 0 : y;
    y = y > _height - 1 ? _height - 1 : y;

    Point2 cur_pt(x, y);

    // std::cout << "Pre : " << m_ptPre.x << " " <<m_ptPre.y << std::endl;
    // std::cout << "Cur : " << cur_pt.x << " " <<cur_pt.y << std::endl;
    if (_iButton == GLUT_LEFT_BUTTON) {
        //_scene->rotate(_ptPre, cur_pt);
        _cur_page += (y - (int)_pt_pre.y);
        if (_cur_page < 0) {
            _cur_page = 0;
        }
        if (_cur_page > _sum_page - 1) {
            _cur_page = _sum_page -1;
        }
        _scene->page_to(_cur_page);

    } else if (_iButton == GLUT_MIDDLE_BUTTON) {
        _scene->pan(_pt_pre, cur_pt);
    } else if (_iButton == GLUT_RIGHT_BUTTON) {
        _scene->zoom(_pt_pre, cur_pt);
    }

    _pt_pre = cur_pt;
    glutPostRedisplay();
}
}

int TE_MPRScene(int argc, char* argv[]) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "opengl" || std::string(argv[1]) == "cuda")) {
            if (std::string(argv[1]) == "opengl") {
                _gpu_platform = GL_BASE;
            }
            else if (std::string(argv[1]) == "cuda") {
                _gpu_platform = CUDA_BASE;
            }
        }

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(_width, _height);

        if (_gpu_platform == GL_BASE) {
            glutCreateWindow("Test GL MPR Scene");
        }
        else {
            glutCreateWindow("Test CUDA MPR Scene");
        }

        if (GLEW_OK != glewInit()) {
            std::cout << "Init glew failed!\n";
            return -1;
        }

        Init();

        GLEnvironment env;
        int major, minor;
        env.get_gl_version(major, minor);

        glutDisplayFunc(Display);
        glutReshapeFunc(Resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        glutMainLoop();
        return 0;
    } catch (const Exception& e) {
        std::cout << e.what();
        return -1;
    }
}