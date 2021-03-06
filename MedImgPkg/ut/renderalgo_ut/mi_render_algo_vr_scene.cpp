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

#include "cudaresource/mi_cuda_resource_manager.h"
#include "cudaresource/mi_cuda_time_query.h"

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
#include "renderalgo/mi_fps.h"

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

using namespace medical_imaging;
namespace {

std::shared_ptr<ImageDataHeader> _data_header;
std::shared_ptr<ImageData> _volume_data;
std::shared_ptr<VolumeInfos> _volumeinfos;

std::shared_ptr<VRScene> _scene;
int _ww;
int _wl;

std::shared_ptr<GLTimeQuery> _time_query;
std::shared_ptr<CudaTimeQuery> _time_query_2;

int _width = 1024;
int _height = 1024;
int _iButton = -1;
Point2 _pt_pre;
int _iTestCode = 0;
bool _pan_status = false;

int _act_label_idx = 0;
std::vector<unsigned char> _vis_labels;

GPUPlatform _gpu_platform = GL_BASE;

MaskMode _mask_mode = MASK_MULTI_LABEL;
CompositeMode _composite_mode = COMPOSITE_DVR;
ShadingMode _shading_mode = SHADING_PHONG;

float _minip_threshold = 100.0f;

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
    FileUtil::get_all_file_recursion(root+"/LIDC-IDRI-1002", dcm_postfix, files);
    return files;
}

void Finalize() {
    _data_header.reset();
    _volume_data.reset();
    _volumeinfos.reset();
    _scene.reset();
    _time_query.reset();
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
    const unsigned int data_len = _volume_data->_dim[0]*_volume_data->_dim[1]*_volume_data->_dim[2];
    printf("spacing : %lf %lf %lf \n" , _volume_data->_spacing[0] , _volume_data->_spacing[1] , _volume_data->_spacing[2]);

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
    std::ifstream in(root+"/mask.raw" , std::ios::in);
    if(in.is_open()) {
        in.read(mask_raw, data_len);
        in.close();
    } else {
        memset(mask_raw , 1 , data_len);
    }

    std::set<unsigned char> target_label_set;
    RunLengthOperator run_length_op;
    std::ifstream in2(root + "/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.rle", std::ios::binary | std::ios::in);
    if (in2.is_open()) {
        in2.seekg(0, in2.end);
        const int code_len = in2.tellg();
        in2.seekg(0, in2.beg);
        unsigned int *code_buffer = new unsigned int[code_len];
        in2.read((char*)code_buffer, code_len);
        in2.close();
        unsigned char* mask_target = new unsigned char[data_len];

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

    _scene.reset(new VRScene(_width, _height, GPU_BASE, _gpu_platform));
    const float PRESET_CT_LUNGS_WW = 1500;
    const float PRESET_CT_LUNGS_WL = -400;
    _ww = PRESET_CT_LUNGS_WW;
    _wl = PRESET_CT_LUNGS_WL;

    _scene->set_volume_infos(_volumeinfos);
    _scene->set_sample_step(0.5);
    _scene->set_global_window_level(_ww, _wl);
    _scene->set_window_level(_ww, _wl, 0);
    _scene->set_window_level(_ww, _wl, 1);
    _scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
    _scene->set_mask_mode(_mask_mode);
    _scene->set_interpolation_mode(LINEAR);
    _scene->set_composite_mode(_composite_mode);
    _scene->set_shading_mode(_shading_mode);
    _scene->set_proxy_geometry(PG_BRICKS);
    _scene->set_test_code(_iTestCode);
    _scene->set_navigator_visibility(true);
    _scene->set_minip_threshold(_minip_threshold);

    // Time query
    _time_query = GLResourceManagerContainer::instance()
                  ->get_time_query_manager()
                  ->create_object("TQ");
    _time_query->initialize();

    _time_query_2 = CudaResourceManager::instance()->create_cuda_time_query("TQ");
    _time_query_2->initialize();

    // Transfer function
    std::shared_ptr<ColorTransFunc> pseudo_color;
#ifdef WIN32
    const std::string pseudo_color_xml = "../../../config/lut/2d/hot_green.xml";
#else
    const std::string pseudo_color_xml = "../config/lut/2d/hot_green.xml";
#endif

    if (IO_SUCCESS !=
            TransferFuncLoader::load_pseudo_color(pseudo_color_xml, pseudo_color)) {
        MI_RENDERALGO_LOG(MI_ERROR) << "load pseudo failed!\n";
    }

    _scene->set_pseudo_color(pseudo_color);

    std::shared_ptr<ColorTransFunc> color;
    std::shared_ptr<OpacityTransFunc> opacity;
    float ww, wl;   
    RGBAUnit background;
    Material material;
#ifdef WIN32
    std::string color_opacity_xml = "../../../config/lut/3d/ct_cta.xml";
#else
    std::string color_opacity_xml = "../config/lut/3d/ct_cta.xml";
#endif

    if (IO_SUCCESS !=
            TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity,
                    ww, wl, background, material)) {
        printf("load lut : %s failed.\n" , color_opacity_xml.c_str());
    }

    _scene->set_color_opacity(color, opacity, 0);
    _scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
    _scene->set_material(material, 0);
    _scene->set_window_level(ww, wl, 0);

    _scene->set_color_opacity(color, opacity, 1);
    _scene->set_material(material, 1);
    _scene->set_window_level(ww, wl, 1);

    _ww = ww;
    _wl = wl;

#ifdef WIN32
    color_opacity_xml = "../../../config/lut/3d/ct_lung_nodule.xml";
#else
    color_opacity_xml = "../config/lut/3d/ct_lung_nodule.xml";
#endif
    if (IO_SUCCESS !=
        TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity,
        ww, wl, background, material)) {
        printf("load lut : %s failed.\n" , color_opacity_xml.c_str());
    }
    std::vector<unsigned char> vis_labels;
    vis_labels.push_back(1);
    printf("target label : ");
    for (auto it = target_label_set.begin(); it != target_label_set.end() ; ++it)
    {
        printf("%i " , int(*it));
        vis_labels.push_back(*it);
        _scene->set_color_opacity(color, opacity, *it);
        _scene->set_material(material, *it);
        _scene->set_window_level(ww, wl, *it);
    }
    printf("\n");
    _scene->set_visible_labels(vis_labels);
    _vis_labels = vis_labels;

    if (_mask_mode == MASK_NONE) {
        _scene->set_global_window_level(ww, wl);
    }
}

void Display() {
    try {

        CHECK_GL_ERROR;

        glViewport(0, 0, _width, _height);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        // static bool ds = false;
        // ds = !ds;
        // _scene->set_downsample(ds);

         //std::shared_ptr<CameraBase> camera = _scene->get_camera();
         //Quat4 q(5.0 / 360.0 * 2.0 * 3.1415926, Vector3(0, 1, 0));
         //camera->rotate(q);

        CHECK_GL_ERROR;

        // _time_query->begin();

        _time_query_2->begin();

        _scene->set_dirty(true);

        CHECK_GL_ERROR;

        _scene->render();

        CHECK_GL_ERROR;

        if (_render_to_back) {
            _scene->render_to_back(); 
        }

        CHECK_GL_ERROR;

        // download result

         glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);

        //_scene->set_downsample(true);
        _scene->download_image_buffer();
        //_scene->set_downsample(false);

        CHECK_GL_ERROR;
        _scene->swap_image_buffer();
        unsigned char* buffer = nullptr;
        int buffer_size = 0;

        CHECK_GL_ERROR;

        _scene->get_image_buffer(buffer, buffer_size);

        CHECK_GL_ERROR;
    
        
        const std::string jpeg_file_name = GL_BASE == _gpu_platform ? "output_ut_gl.jpeg" : "output_ut_cuda.jpeg";

#ifdef WIN32
        FileUtil::write_raw("D:/temp/" + jpeg_file_name, buffer, buffer_size);
#else
        FileUtil::write_raw("/home/wangrui22/data" + jpeg_file_name, buffer, buffer_size);
#endif
      
         //const float render_time  = _time_query->end();
         const float render_time = _time_query_2->end();
         //MI_RENDERALGO_LOG(MI_INFO) << "rendering time : " << render_time << " " << buffer_size;

        int fps = _fps.fps(render_time);
        static int tiker = 0;
        tiker++;
        if (tiker > 50) {
            MI_RENDERALGO_LOG(MI_INFO) << "FPS:  " << fps;
            tiker = 0;
        }

        glutSwapBuffers();
    } catch (Exception& e) {
        MI_RENDERALGO_LOG(MI_ERROR) << e.what();
        abort();
    }
}

void Keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case '[':{
        _minip_threshold += 10.0f;
        std::cout << "MinIP threshold: " << _minip_threshold << std::endl;
        _scene->set_minip_threshold(_minip_threshold);
        break;
    }
    case ']': {
        _minip_threshold -= 10.0f;
        std::cout << "MinIP threshold: " << _minip_threshold << std::endl;
        _scene->set_minip_threshold(_minip_threshold);
        break;
    }
    case 'm': {
        static int s_mode = 0;
        s_mode++;
        if (s_mode > 3 ) {
            s_mode = 0; 
        }

        if (s_mode == 0) {
            std::cout << "DVR \n";
            _composite_mode = COMPOSITE_DVR;
            _mask_mode = MASK_MULTI_LABEL;
            _shading_mode = SHADING_PHONG;
            _scene->set_composite_mode(_composite_mode);
            _scene->set_mask_mode(_mask_mode);
            _scene->set_shading_mode(_shading_mode);
        }
        else if (s_mode == 1) {
            std::cout << "MIP \n";
            _composite_mode = COMPOSITE_MIP;
            _mask_mode = MASK_MULTI_LABEL;
            _shading_mode = SHADING_NONE;
            _scene->set_composite_mode(_composite_mode);
            _scene->set_mask_mode(_mask_mode);
            _scene->set_shading_mode(_shading_mode);
        } else if (s_mode == 2) {
            std::cout << "MINIP \n";
            _composite_mode = COMPOSITE_MINIP;
            _mask_mode = MASK_MULTI_LABEL;
            _shading_mode = SHADING_NONE;
            _scene->set_composite_mode(_composite_mode);
            _scene->set_mask_mode(_mask_mode);
            _scene->set_shading_mode(_shading_mode);
        } else if (s_mode == 3) {
            std::cout << "AVERAGE \n";
            _composite_mode = COMPOSITE_AVERAGE;
            _mask_mode = MASK_MULTI_LABEL;
            _shading_mode = SHADING_NONE;
            _scene->set_composite_mode(_composite_mode);
            _scene->set_mask_mode(_mask_mode);
            _scene->set_shading_mode(_shading_mode);
        }
        break;
    }
    case 'b': {
        _render_to_back  = !_render_to_back;
        break;
    }
    case 'f': {
        _iTestCode = 1 - _iTestCode;
        _scene->set_test_code(_iTestCode);
        break;
    }
    case 'i': {
        std::cout << "\n<><><><><><><><><><> GL RESOURCE <><><><><><><><><><>\n";
        std::cout << GLResourceManagerContainer::instance()->get_specification("\n");
        std::cout << "<><><><><><><><><><> GL RESOURCE <><><><><><><><><><>\n";
        std::cout << "\n<><><><><><><><><><> CUDA RESOURCE <><><><><><><><><><>\n";
        std::cout << CudaResourceManager::instance()->get_specification("\n");
        std::cout << "<><><><><><><><><><> CUDA RESOURCE <><><><><><><><><><>\n";
        break;
    }
    case 'r': {
        _scene.reset(new VRScene(_width, _height, GPU_BASE, _gpu_platform));
        const float PRESET_CT_LUNGS_WW = 1500;
        const float PRESET_CT_LUNGS_WL = -400;
        _ww = PRESET_CT_LUNGS_WW;
        _wl = PRESET_CT_LUNGS_WL;

        _scene->set_volume_infos(_volumeinfos);
        _scene->set_sample_step(1.0);
        _scene->set_global_window_level(_ww, _wl);
        _scene->set_window_level(_ww, _wl, 0);
        _scene->set_composite_mode(COMPOSITE_MIP);
        _scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        _scene->set_mask_mode(MASK_NONE);
        _scene->set_interpolation_mode(LINEAR);
        _scene->set_proxy_geometry(PG_CUBE);
        _scene->set_test_code(_iTestCode);
        _scene->initialize();
        _scene->set_display_size(_width, _height);
        break;
    }
    case 'a' : {
            _act_label_idx += 1;
            if (_act_label_idx > (int)_vis_labels.size() - 1) {
                _act_label_idx = 0;
            }
            unsigned char act_label = _vis_labels[_act_label_idx];
            printf("act labels is : n%i\n" , (int)act_label);
            float ww , wl;
            _scene->get_window_level(ww , wl, act_label);
            _ww = (int)ww;
            _wl = (int)wl;
            break;
        }
    case 'h' : {
        static int lut_id = 0;
        static const int LUT_NUM = 11;
        static const std::string LUT_LIST[LUT_NUM] = {
            "../config/lut/3d/ct_cta.xml",
            "../config/lut/3d/ct_cta_1.xml",
            "../config/lut/3d/ct_lung_glass.xml",
            "../config/lut/3d/ct_lung_glass_2.xml",
            "../config/lut/3d/ct_calcification.xml",
            "../config/lut/3d/ct_carotids.xml",
            "../config/lut/3d/ct_clr_abd_aorta_1.xml",
            "../config/lut/3d/ct_clr_abd_aorta_2.xml",
            "../config/lut/3d/ct_clr_carotid_1.xml",
            "../config/lut/3d/ct_clr_carotid_2.xml",
            "../config/lut/3d/ct_color_vessel_gd.xml"   
        };
        lut_id += 1;
        if (lut_id > LUT_NUM-1) {
            lut_id = 0;
        }
        const std::string color_opacity_xml = LUT_LIST[lut_id];
        std::shared_ptr<ColorTransFunc> color;
        std::shared_ptr<OpacityTransFunc> opacity;
        float ww, wl;
        RGBAUnit background;
        Material material;
        if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity,
                        ww, wl, background, material)) {
            printf("load lut : %s failed.\n" , color_opacity_xml.c_str());
        }
    
        _scene->set_color_opacity(color, opacity, 0);
        _scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
        _scene->set_material(material, 0);
        _scene->set_window_level(ww, wl, 0);
    
        _scene->set_color_opacity(color, opacity, 1);
        _scene->set_material(material, 1);
        _scene->set_window_level(ww, wl, 1);
        break;
    }
    case 'p' : {
        _pan_status = !_pan_status;
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

    if (status == GLUT_DOWN) {
        //_scene->set_expected_fps(300);
        //_scene->set_downsample(true);
    } else if (status == GLUT_UP) {
        //_scene->set_downsample(false);
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

    if (_iButton == GLUT_LEFT_BUTTON) {
        _scene->rotate(_pt_pre, cur_pt);

    } else if (_iButton == GLUT_MIDDLE_BUTTON) {
        if (_pan_status) {
            _scene->pan(_pt_pre , cur_pt);
        } else {
            if (_composite_mode == COMPOSITE_MIP || _composite_mode == COMPOSITE_MINIP || _composite_mode == COMPOSITE_AVERAGE) {
                float ww, wl;
                _scene->get_global_window_level(ww, wl);
                ww += (x - (int)_pt_pre.x);
                wl += ((int)_pt_pre.y - y);
                ww = ww < 0 ? 1 : ww;
                _scene->set_global_window_level(ww, wl);
            } else {
                _ww += (x - (int)_pt_pre.x);
                _wl += ((int)_pt_pre.y - y);
                _ww = _ww < 0 ? 1 : _ww;
                _scene->set_window_level(_ww, _wl, _vis_labels[_act_label_idx]);
            }
            MI_RENDERALGO_LOG(MI_DEBUG) << "wl : " << _ww << " " << _wl << std::endl;
            
        }

    } else if (_iButton == GLUT_RIGHT_BUTTON) {
        _scene->zoom(_pt_pre, cur_pt);
    }

    _pt_pre = cur_pt;
    glutPostRedisplay();
}
}

int TE_VRScene(int argc, char* argv[]) {
#ifndef WIN32
    chdir(dirname(argv[0]));
#endif

    try {

        if (argc == 2 &&( std::string(argv[1]) == "opengl" || std::string(argv[1]) == "cuda")) {
            if (std::string(argv[1]) == "opengl") {
                _gpu_platform = GL_BASE;
            } else if (std::string(argv[1]) == "cuda") {
                _gpu_platform = CUDA_BASE;
            }
        }

        //Init();

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(_width, _height);

        if (_gpu_platform == GL_BASE) {
            glutCreateWindow("Test GL VR Scene");
        } else {
            glutCreateWindow("Test CUDA VR Scene");
        }
        

        if (GLEW_OK != glewInit()) {
            MI_RENDERALGO_LOG(MI_FATAL) << "Init glew failed!\n";
            return -1;
        }

        GLEnvironment env;
        int major, minor;
        env.get_gl_version(major, minor);

        glutDisplayFunc(Display);
        glutReshapeFunc(Resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        Init();

        glutMainLoop();

        //Logger::instance()->finalize();

        Finalize();

        return 0;
    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << e.what();
        return -1;
    }
}