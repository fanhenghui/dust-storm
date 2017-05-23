#include <iostream>

#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/camera_interactor.h"

#include "neural_net.h"
#include "gesture_drawer.h"
#include "gesture_recogizer.h"
#include "param.h"

namespace
{
    int _width = 800;
    int _height = 800;
    int _pre_btn = 0;
    int _pre_btn_state = 0;
    Point2 _pre_pos;


    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;
    std::shared_ptr<GestureDrawer> _gestuer_drawer;
    std::shared_ptr<GestureRecogizer> _gestuer_recogizer;

    void init()
    {
        _camera.reset(new OrthoCamera());
        _camera->set_eye(Point3(0, 0, 10.0));
        _camera->set_look_at(Point3::S_ZERO_POINT);
        _camera->set_up_direction(Vector3(0, 1, 0));
        _camera->set_ortho(-1, 1, -1, 1, 5, 15);

        _camera_interactor.reset(new OrthoCameraInteractor(_camera));

        _gestuer_drawer.reset(new GestureDrawer());

        _gestuer_recogizer.reset(new GestureRecogizer());
        _gestuer_recogizer->train();

    }

    void display()
    {
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, _width, _height);

        _gestuer_drawer->draw();

        glutSwapBuffers();

    }

    void mouse_click(int btn, int state, int x, int y)
    {
        _pre_btn = btn;
        _pre_btn_state = state;

        if (btn == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
        {
            _gestuer_drawer->clear();
            _gestuer_drawer->push_back(Point2( (x/(double)_width - 0.5)*2 ,( ( _height -y) / (double)_height - 0.5) * 2));
        }
        else if (btn == GLUT_LEFT_BUTTON &&state == GLUT_UP)
        {
            _gestuer_drawer->calculate_splits();
            std::vector<Point2> pts = _gestuer_drawer->get_splits();
            std::vector<double> pt_array;
            pt_array.reserve(pts.size() * 2);
            for (int i = 0 ; i<pts.size() ; ++i)
            {
                pt_array.push_back(pts[i].x);
                pt_array.push_back(pts[i].y);
            }
            const std::string pattern_name = _gestuer_recogizer->get_gesture_pattern_name(pt_array);
            std::cout << "Pattern name : " << pattern_name << std::endl;
        }

        _pre_pos = Point2(x, y);

        glutPostRedisplay();
    }

    void mouse_motion(int x, int y)
    {
        if (_pre_btn == GLUT_LEFT_BUTTON)
        {
            _gestuer_drawer->push_back(Point2((x / (double)_width - 0.5) * 2, ((_height - y) / (double)_height - 0.5) * 2));
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



void main_mgr(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(_width, _height);

    glutCreateWindow("Mouse gesture recognize");

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