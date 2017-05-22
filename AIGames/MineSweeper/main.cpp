#include <iostream>
#include <fstream>
#include <thread>
#include <ctime>
#include <chrono>


#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/camera_interactor.h"
#include "Core/gl_texture_2d.h"
#include "Core/gl_resource_manager_container.h"
#include "Core/targa.h"
#include "Core/gl_utils.h"

#include "mine.h"
#include "mine_sweeper.h"
#include "battle_space.h"
#include "param.h"

#define PNG_READ_SUPPORTED

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

    std::shared_ptr<BattleSpace> _battle_space;

    void run()
    {
        while (true)
        {

            _battle_space->update();

        }
    }

    void init()
    {
        _camera.reset(new OrthoCamera());
        _camera->set_eye(Point3(0, 0, 10.0));
        _camera->set_look_at(Point3::S_ZERO_POINT);
        _camera->set_up_direction(Vector3(0, 1, 0));
        _camera->set_ortho(-1, 1, -1, 1, 5, 15);

        _camera_interactor.reset(new OrthoCameraInteractor(_camera));

        _battle_space.reset(new BattleSpace());
        _battle_space->register_render_callback(glutPostRedisplay);

        std::thread th(run);
        th.detach();
    }

    void display()
    {
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, _width, _height);

        //_battle_space->update();
        _battle_space->render();

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
            _battle_space->enable_fast_training(!_battle_space->is_fast_training());
            break;
        }
        default:
            break;
        }

        glutPostRedisplay();
    }

    void idle()
    {
        //glutPostRedisplay();
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
    glutIdleFunc(idle);

    init();

    glutMainLoop();
}