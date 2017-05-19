#include <iostream>

#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/camera_interactor.h"


#define CROSSOVER_RATE 0.7
#define MUTATION_RATE 0.001
#define POP_SIZE 140
#define CHROMO_LENGTH 70

namespace
{
    int _width = 800;
    int _height = 800;
    int _pre_btn = 0;
    int _pre_btn_state = 0;
    Point2 _pre_pos;

    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

    void init()
    {
        _camera.reset(new OrthoCamera());
        _camera->set_eye(Point3(0, 0, 10.0));
        _camera->set_look_at(Point3::S_ZERO_POINT);
        _camera->set_up_direction(Vector3(0, 1, 0));
        _camera->set_ortho(-1, 1, -1, 1, 5, 15);

        _camera_interactor.reset(new OrthoCameraInteractor(_camera));

    }

    void display()
    {
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, _width, _height);

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

    glutCreateWindow("TSP");

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