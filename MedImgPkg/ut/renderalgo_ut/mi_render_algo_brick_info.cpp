#include "GL/glew.h"

#include "util/mi_file_util.h"
#include "log/mi_logger.h"

#include "arithmetic/mi_run_length_operator.h"

#include "io/mi_configure.h"
#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "glresource/mi_gl_utils.h"

#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_brick_pool.h"
#include "renderalgo/mi_render_algo_logger.h"

#ifdef WIN32
#include "GL/glut.h"
#else
#include "GL/freeglut.h"
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#endif

using namespace medical_imaging;

int _width = 512;
int _height = 512;

std::shared_ptr<ImageDataHeader> _data_header;
std::shared_ptr<ImageData> _volume_data;
std::shared_ptr<VolumeInfos> _volumeinfos_gl;
std::shared_ptr<VolumeInfos> _volumeinfos_cuda;
std::vector<unsigned char> _visible_labels;

#ifdef WIN32
const std::string root = "E:/data/MyData/demo/lung/";
#else
const std::string root = "/home/wangrui22/data/demo/lung/";
#endif

static std::vector<std::string> GetFiles() {
    std::vector<std::string> files;
    std::set<std::string> dcm_postfix;
    dcm_postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(root + "/LIDC-IDRI-1002", dcm_postfix, files);
    return files;
}

static void initialize() {
    Configure::instance()->set_processing_unit_type(GPU);
    GLUtils::set_check_gl_flag(true);
#ifdef WIN32
    Logger::instance()->bind_config_file("./config/log_config");
#else
    Logger::instance()->bind_config_file("../config/log_config");
#endif

    Logger::instance()->initialize();

    //Volume
    std::vector<std::string> files = GetFiles();
    DICOMLoader loader;
    loader.load_series(files, _volume_data, _data_header);
    const unsigned int data_len = _volume_data->_dim[0] * _volume_data->_dim[1] * _volume_data->_dim[2];

    //Mask
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
    target_label_set.insert(1);
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
    
    _visible_labels.clear();
    std::cout << "target label: ";
    for (auto it = target_label_set.begin(); it != target_label_set.end(); ++it) {
        std::cout << int(*it) << " ";
        _visible_labels.push_back(*it);
    }
    std::cout << std::endl;

    FileUtil::write_raw(root + "/target_mask.raw", mask_raw, data_len);



    _volumeinfos_gl.reset(new VolumeInfos(GPU_BASE, GL_BASE));
    _volumeinfos_gl->set_data_header(_data_header);
    _volumeinfos_gl->set_volume(_volume_data);
    _volumeinfos_gl->set_mask(mask_data);
    _volumeinfos_gl->get_brick_pool()->add_visible_labels_cache(_visible_labels);
    _volumeinfos_gl->refresh();

    _volumeinfos_cuda.reset(new VolumeInfos(GPU_BASE, CUDA_BASE));
    _volumeinfos_cuda->set_data_header(_data_header);
    _volumeinfos_cuda->set_volume(_volume_data);
    _volumeinfos_cuda->set_mask(mask_data);
    _volumeinfos_cuda->get_brick_pool()->add_visible_labels_cache(_visible_labels);
    _volumeinfos_cuda->refresh();
}

static void finalize() {

}

static void compare_brick_info() {
    std::shared_ptr<BrickPool> gl_brick_pool = _volumeinfos_gl->get_brick_pool();
    std::shared_ptr<BrickPool> cuda_brick_pool = _volumeinfos_cuda->get_brick_pool();

    const unsigned int brick_count = cuda_brick_pool->get_brick_count();

    VolumeBrickInfo* gl_volume_brick_info = gl_brick_pool->get_volume_brick_info();
    VolumeBrickInfo* cuda_volume_brick_info = cuda_brick_pool->get_volume_brick_info();

    //gl_brick_pool->debug_save_volume_brick_info("d:/temp/gl_volume_brick_info.txt");
    //cuda_brick_pool->debug_save_volume_brick_info("d:/temp/cuda_volume_brick_info.txt");
    bool success = true;
    for (unsigned int i = 0; i < brick_count; ++i) {
        if (fabs(gl_volume_brick_info[i].min - cuda_volume_brick_info[i].min) > FLOAT_EPSILON ||
            fabs(gl_volume_brick_info[i].max - cuda_volume_brick_info[i].max) > FLOAT_EPSILON) {
            success = false;
            break;
        }
    }

    std::cout << "compare volume brick info: " << success << std::endl;

    MaskBrickInfo* gl_mask_brick_info = gl_brick_pool->get_mask_brick_info(_visible_labels);
    MaskBrickInfo* cuda_mask_brick_info = cuda_brick_pool->get_mask_brick_info(_visible_labels);
    success = true;
    for (unsigned int i = 0; i < brick_count; ++i) { 
        if (gl_mask_brick_info[i].label != cuda_mask_brick_info[i].label) {
            success = false;
            break;
        }
    }
    std::cout << "compare mask brick info: " << success << std::endl;
    gl_brick_pool->debug_save_mask_brick_infos("d:/temp/gl");
    cuda_brick_pool->debug_save_mask_brick_infos("d:/temp/cuda");
}

static void display() {
    glViewport(0, 0, _width, _height);
    glClearColor(1.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glutSwapBuffers();
}

int TE_BrickInfo(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(_width, _height);

    glutCreateWindow("Test Brick Info");

    if (GLEW_OK != glewInit()) {
        MI_RENDERALGO_LOG(MI_FATAL) << "Init GLEW failed!\n";
        return -1;
    }

    glutDisplayFunc(display);

    initialize();

    compare_brick_info();

    glutMainLoop();

    finalize();

    return 0;
}