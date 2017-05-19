#include <iostream>

#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/camera_interactor.h"

#include "maze.h"
#include "maze_drawer.h"
#include "my_maze.h"
#include "maze_solver.h"


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

    std::shared_ptr<Maze> _maze;
    std::shared_ptr<MazeDrawer> _maze_drawer;
    std::shared_ptr<MazeSolver> _maze_solver;
    const int _iteration = 100;
    int _cur_iteration = 100;

    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

    void init()
    {
        _maze.reset(new Maze((&MY_MAZE[0][0]), MAZE_ROW, MAZE_COLUMN));
        
        _camera.reset(new OrthoCamera());
        _camera->set_eye(Point3(0,0,10.0));
        _camera->set_look_at(Point3::S_ZERO_POINT);
        _camera->set_up_direction(Vector3(0, 1, 0));
        _camera->set_ortho(-1, 1, -1, 1, 5, 15);

        _maze_drawer.reset(new MazeDrawer());
        _maze_drawer->set_maze(_maze);
        _maze_drawer->set_camera(_camera);

        _camera_interactor.reset(new OrthoCameraInteractor(_camera));

        _maze_solver.reset(new MazeSolver(CROSSOVER_RATE, MUTATION_RATE, POP_SIZE, CHROMO_LENGTH, 2));
        _maze_solver->set_maze(_maze);
    }

    void display()
    {
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, _width, _height);
        _maze_drawer->draw();

        glutSwapBuffers();

        if (_cur_iteration < _iteration)
        {
            if (_cur_iteration == 0)
            {
                _maze_solver->create_start_population();
            }

            std::cout << _cur_iteration << std::endl;
            _maze_solver->epoch();
            std::vector<int> route = _maze_solver->decode(_maze_solver->get_fittest_chromosome());
            _maze_drawer->set_route(route);
            ++_cur_iteration;

            //Check solved in advance
            int end_row, end_column;
            if (_maze->test_route(route, end_row , end_column))
            {
                _cur_iteration = _iteration;
            }

            glutPostRedisplay();
        }
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
            _cur_iteration = 0;
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

    glutCreateWindow("Maze");

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

    _maze->print();

    glutMainLoop();
}