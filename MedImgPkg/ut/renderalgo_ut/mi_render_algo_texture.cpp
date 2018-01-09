#include "GL/glew.h"

#include "util/mi_file_util.h"
#include "log/mi_logger.h"

#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_resource_manager_container.h"

#include "cudaresource/mi_cuda_gl_texture_2d.h"
#include "cudaresource/mi_cuda_resource_manager.h"

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

static int _width = 1024;
static int _height = 1024;

static std::vector<GLTexture2DPtr> _gl_tex_2ds;
static std::vector<CudaGLTexture2DPtr> _cuda_gl_tex_2ds;
static int _sum = 300;

static void initialize() {
    GLUtils::set_check_gl_flag(true);
}

static void finalize() {

}

static void test_cuda_gl_texture() {
    unsigned char* _checkboard_data = new unsigned char[_width*_height * 4];
    int tag_x = 0;
    int tag_y = 0;
    int idx = 0;
    for (int y = 0; y < _height; ++y) {
        for (int x = 0; x < _width; ++x) {
            tag_x = x / 32;
            tag_y = y / 32;
            idx = y*_width + x;
            if ((tag_x + tag_y) % 2 == 0) {
                _checkboard_data[idx * 4] = 200;
                _checkboard_data[idx * 4 + 1] = 200;
                _checkboard_data[idx * 4 + 2] = 200;
                _checkboard_data[idx * 4 + 3] = 255;
            }
            else {
                _checkboard_data[idx * 4] = 20;
                _checkboard_data[idx * 4 + 1] = 20;
                _checkboard_data[idx * 4 + 2] = 20;
                _checkboard_data[idx * 4 + 3] = 255;
            }
        }
    }
    unsigned char* download = new unsigned char[_width*_height * 4];
    for (int i = 0; i < _sum; ++i) {
        GLTexture2DPtr tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object("test");
        tex->initialize();
        tex->bind();
        tex->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, _checkboard_data);
        //tex->download(GL_RGBA, GL_UNSIGNED_BYTE, download);
        //FileUtil::write_raw("D:/temp/check_board.rgba", download, _width*_height * 4);
        tex->unbind();
        _gl_tex_2ds.push_back(tex);
    }

    for (int i=0; i<_sum; ++i) {
        CudaGLTexture2DPtr tex = CudaResourceManager::instance()->create_cuda_gl_texture_2d("cuda test");
        tex->register_gl_texture(_gl_tex_2ds[i], cudaGraphicsRegisterFlagsReadOnly);
        _gl_tex_2ds[i]->bind();
        tex->map_gl_texture();
        if (i == 0) {
            tex->download(_width*_height * 4, download);
            FileUtil::write_raw("D:/temp/check_board_cuda.rgba", download, _width*_height * 4);
        }
        tex->unmap_gl_texture();
        _gl_tex_2ds[i]->unbind();
        _cuda_gl_tex_2ds.push_back(tex);
    }

    std::cout << "DONE.";
}

static void display() {
    glViewport(0, 0, _width, _height);
    glClearColor(1.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glutSwapBuffers();
}

int TE_Texture(int argc, char* argv[]) {
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

    test_cuda_gl_texture();

    glutMainLoop();

    finalize();

    return 0;
}