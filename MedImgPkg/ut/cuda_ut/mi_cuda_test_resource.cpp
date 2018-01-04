#include <GL/glew.h>
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

#include "log/mi_logger.h"
#include "cudaresource/mi_cuda_texture_1d.h"
#include "cudaresource/mi_cuda_texture_2d.h"
#include "cudaresource/mi_cuda_texture_3d.h"
#include "cudaresource/mi_cuda_gl_texture_2d.h"
#include "cudaresource/mi_cuda_device_memory.h"
#include "cudaresource/mi_cuda_resource_manager.h"
#include <memory>

using namespace medical_imaging;

extern "C"
void draw_slice(int i, cudaTextureObject_t tex, dim3 dim, int width, int height, unsigned char* d_rgba);

int _width = 1024;
int _height = 1024;
unsigned char * _h_canvas = nullptr;
unsigned char * _d_canvas = nullptr;

int _dim[3] = { 512,512,512};
std::shared_ptr<CudaTexture3D>  _cuda_volume_tex;
cudaTextureObject_t _volume_tex_obj;
cudaTextureObject_t _volume_tex_obj_norm;

int _cur_slice = 0;

static void init() {
#ifdef WIN32
    Logger::instance()->bind_config_file("./config/log_config");
#else
    Logger::instance()->bind_config_file("../config/log_config");
#endif
    MI_LOG(MI_INFO) << "OK";
    _h_canvas = new unsigned char[_width*_height*4];
    cudaMalloc(&_d_canvas, _width*_height * 4);

    
    unsigned short* array = new unsigned short[_dim[0] * _dim[1] * _dim[2]];
    unsigned short* tmp = array;
    for (int z = 0; z < _dim[2]; ++z) {
        tmp = array + z*_dim[0] * _dim[1];
        for (int j = 0; j < _dim[0] * _dim[1]; ++j) {
            tmp[j] = z;
        }
    }

    _cuda_volume_tex = CudaResourceManager::instance()->create_cuda_texture_3d("volume");
    _cuda_volume_tex->load(16, 0, 0, 0, cudaChannelFormatKindUnsigned, _dim[0], _dim[1], _dim[2], array);
    _volume_tex_obj = _cuda_volume_tex->get_object(cudaAddressModeClamp, cudaFilterModePoint, cudaReadModeElementType, false);
    _volume_tex_obj_norm = _cuda_volume_tex->get_object(cudaAddressModeClamp, cudaFilterModePoint, cudaReadModeNormalizedFloat, false);
    delete [] array;

    MI_LOG(MI_INFO) << *CudaResourceManager::instance();
    MI_LOG(MI_INFO) << CudaResourceManager::instance()->get_specification("\n");
}

static void display() {
    glViewport(0,0,_width, _height);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    _cur_slice ++;
    if (_cur_slice > _dim[2] - 1) {
        _cur_slice = 0;
    }
    draw_slice(_cur_slice, _volume_tex_obj_norm, dim3(_dim[0], _dim[1], _dim[2]), _width, _height, _d_canvas);
    cudaMemcpy(_h_canvas, _d_canvas, _width*_height*4, cudaMemcpyDeviceToHost);
    
    glDrawPixels(_width, _height, GL_RGBA, GL_UNSIGNED_BYTE, _h_canvas);

    glutSwapBuffers();
}

static void keyboard(unsigned char key, int x, int y) {
    switch (key)
    {
    case 'a': {
        static unsigned short val = 300;
        if (val == 0) {
            val = 300;
        } else {
            val = 0;
        }
        unsigned short* array = new unsigned short[_dim[0]/3 * _dim[1]/3 * _dim[2]];
        for (unsigned int i = 0; i < _dim[0] / 3 * _dim[1] / 3 * _dim[2]; ++i) {
            array[i] = val;
        }
        _cuda_volume_tex->update(_dim[0] / 3, _dim[1] / 3 , 0, _dim[0] / 3, _dim[1] / 3 , _dim[2], array);
        delete [] array;
        break;
    }

    default:
        break;
    }
}

static void idle() {
    glutPostRedisplay();
}

int cuda_test_resource(int argc, char* argv[]) {

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(_width, _height);

    glutCreateWindow("Simple Ray Tracing");

    if (GLEW_OK != glewInit()) {
        std::cout << "Init GLEW failed!\n";
        return -1;
    }

    init();

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);

    glutMainLoop();

    return 0;

}