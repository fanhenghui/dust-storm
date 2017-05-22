
#include <iostream>
#include <fstream>

#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/camera_interactor.h"
#include "Core/gl_texture_2d.h"
#include "Core/gl_resource_manager_container.h"
#include "Core/targa.h"
#include "Core/gl_utils.h"
#include "Ext/libpng15/png.h"
#include "Ext/libpng15/pngconf.h"

#include "mine.h"
#include "mine_sweeper.h"

namespace
{
    int _width = 800;
    int _height = 800;
    int _pre_btn = 0;
    int _pre_btn_state = 0;
    Point2 _pre_pos;


    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

    std::shared_ptr<GLTexture2D> _tex_tank;
    std::shared_ptr<GLTexture2D> _tex_mine;

    Mine _mine;
    MineSweeper _mine_sweeper;

    unsigned char* read_png(const char* filename)
    {
        int i, j;
        int m_width, m_height;
        png_infop info_ptr;             //图片信息的结构体
        png_structp png_ptr;         //初始化结构体，初始生成，调用api时注意传入

        FILE* file = fopen(filename, "rb");    //打开的文件名

        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);   //创建初始化libpng库结构体
        info_ptr = png_create_info_struct(png_ptr);                                                 //创建图片信息结构体

        setjmp(png_jmpbuf(png_ptr));                              //设置错误的返回点

                                                                  // 这句很重要
        png_init_io(png_ptr, file);         //把文件加载到libpng库结构体中

                                            // 读文件了
        png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_EXPAND, 0);        //读文件内容到info_ptr中


                                                                         // 得到文件的宽高色深
        if ((png_ptr != NULL) && (info_ptr != NULL))
        {
            m_width = png_get_image_width(png_ptr, info_ptr);
            m_height = png_get_image_height(png_ptr, info_ptr);                          //通过png库中的api获取图片的宽度和高度

            printf("%s, %d, m_width =%d, m_height = %d\n", __FUNCTION__, __LINE__, m_width, m_height);
        }
        int color_type = png_get_color_type(png_ptr, info_ptr);                          //通过api获取color_type

        printf("%s, %d, color_type = %d\n", __FUNCTION__, __LINE__, color_type);


        int size = m_height * m_width * 4;

        unsigned char *bgra = new unsigned char[size];
        int pos = 0;

        // row_pointers里边就是传说中的rgb数据了

        png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

        return (unsigned char*)row_pointers;

    }
    void init()
    {

        _camera.reset(new OrthoCamera());
        _camera->set_eye(Point3(0, 0, 10.0));
        _camera->set_look_at(Point3::S_ZERO_POINT);
        _camera->set_up_direction(Vector3(0, 1, 0));
        _camera->set_ortho(-1, 1, -1, 1, 5, 15);

        _camera_interactor.reset(new OrthoCameraInteractor(_camera));

        UIDType id;
        _tex_tank = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(id);
        _tex_mine = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(id);

        int w(120), h(120) , component;
        GLenum format;
        unsigned char* img_array = read_png("../../Resource/tank.png");
        glEnable(GL_TEXTURE_2D);
        _tex_tank->initialize();
        _tex_tank->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _tex_tank->load(GL_RGBA8, 120, 120, GL_RGBA, GL_UNSIGNED_BYTE, img_array);
        _tex_tank->unbind();

        delete[] img_array;
        img_array = read_png("../../Resource/mine.png");
        _tex_mine->initialize();
        _tex_mine->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        //unsigned char* temp = new unsigned char[120 * 120*3];
        //for (int i = 0 ; i< 120*120 ; ++i)
        //{
        //    temp[i * 3] = 255;
        //    temp[i * 3+1] = 0;
        //    temp[i * 3+2] = 255;
        //    //temp[i * 4+3] = 255;
        //}
        _tex_mine->load(GL_RGBA8, 120, 120, GL_RGBA, GL_UNSIGNED_BYTE, img_array);
        //_tex_mine->unbind();
        delete[] img_array;

        _mine.set_texture(_tex_mine);
        //_mine_sweeper.set_texture(_tex_tank);
    }

    void display()
    {
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, _width, _height);

        _mine.draw();

        glutSwapBuffers();
    }

    void mouse_click(int btn, int state, int x, int y)
    {
        _pre_btn = btn;
        _pre_btn_state = state;

        _pre_pos = Point2(x, y);
    }

    void mouse_motion(int x, int y)
    {
        if (_pre_btn == GLUT_LEFT_BUTTON)
        {
            _camera_interactor->rotate(_pre_pos, Point2(x, y), _width, _height);
        }

        _pre_pos = Point2(x, y);

        glutPostRedisplay();
    }

    void reshape(int x, int y)
    {
        _width = x;
        _height = y;
        _camera_interactor->resize(_width, _height);
        glutPostRedisplay();
    }

    void keyboard(unsigned char key, int x, int y)
    {
        switch (key)
        {
        case 'b':
        {
            break;
        }
        default:
            break;
        }

        glutPostRedisplay();
    }
}



void main(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(_width, _height);

    glutCreateWindow("MineSweeper");

    if (GLEW_OK != glewInit())
    {
        std::cout << "Init glew failed!\n";
        return;
    }

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse_click);
    glutMotionFunc(mouse_motion);
    glutKeyboardFunc(keyboard);

    init();

    glutMainLoop();
}