#include <iostream>

#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/camera_interactor.h"

#include "tsp_map.h"
#include "tsp_map_drawer.h"
#include "tsp_solver.h"

#define CROSSOVER_RATE 0.75
#define MUTATION_RATE 0.2
#define POP_SIZE 200

#define CITY_RADIUS 200
#define CITY_NUM 100

#define ITERATION 50000


namespace
{
    int _width = 800;
    int _height = 800;
    int _pre_btn = 0;
    int _pre_btn_state = 0;
    Point2 _pre_pos;

    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

    std::shared_ptr<TSPMap> _tsp_map;
    std::shared_ptr<TSPMapDrawer> _tsp_map_drawer;
    std::shared_ptr<TSPSolver> _tsp_solver;
    const int _iteration = ITERATION;
    int _cur_iteration = ITERATION;

    std::mutex _mutex;

    void generation()
    {
        while (true)
        {
            std::cout << "s" << std::endl;
            while (_cur_iteration < ITERATION)
            {
                std::cout << _cur_iteration << std::endl;
                if (_cur_iteration == 0)
                {
                    _tsp_solver->create_start_population();
                }
                

                //std::cout << _cur_iteration << std::endl;
                const int loop = 1;
                for (int i = 0; i < loop; ++i)
                {
                    _tsp_solver->epoch();
                    std::vector<int> route = _tsp_solver->get_fittest_chromosome()._city_tours;
                    _tsp_map_drawer->set_route(route);
                    ++_cur_iteration;
                }

                //Check solved in advance
                if (abs(_tsp_solver->get_shortest_route() - _tsp_map->get_best_possible_route()) < 0.01)
                {
                    _cur_iteration = _iteration;
                    std::cout << "Iteration : " << _cur_iteration << "Shortest : " << _tsp_solver->get_shortest_route() << "  Best : " << _tsp_map->get_best_possible_route() << std::endl;
                }

                //std::cout << "Shortest : " << _tsp_solver->get_shortest_route() << "  Best : " << _tsp_map->get_best_possible_route() << std::endl;
            }
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

        _tsp_map.reset(new TSPMap(CITY_RADIUS, CITY_NUM));
        _tsp_map_drawer.reset(new TSPMapDrawer());
        _tsp_map_drawer->set_tps_map(_tsp_map);
        _tsp_map_drawer->set_camera(_camera);

        _tsp_solver.reset(new TSPSolver(CROSSOVER_RATE , MUTATION_RATE , POP_SIZE, CITY_NUM));
        _tsp_solver->set_tsp_map(_tsp_map);

        std::thread th(generation);
        th.detach();
    }

    void display()
    {
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, _width, _height);

        _tsp_map_drawer->draw();

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
        else if (_pre_btn == GLUT_RIGHT_BUTTON)
        {
            _camera_interactor->zoom(_pre_pos, Point2(x, y), _width, _height);
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
            std::cout << _cur_iteration << std::endl;
            break;
        }
        default:
            break;
        }

        glutPostRedisplay();
    }

    void idle()
    {
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
    glutIdleFunc(idle);

    init();

    glutMainLoop();
}