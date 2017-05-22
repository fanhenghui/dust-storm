#include "tsp_map_drawer.h"
#include "tsp_map.h"

#include "Core/camera_base.h"
#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"

#define SPHERE_RADIUS 1.5


TSPMapDrawer::TSPMapDrawer()
{
}

TSPMapDrawer::~TSPMapDrawer()
{

}

void TSPMapDrawer::set_tps_map(std::shared_ptr<TSPMap> tsp_map)
{
    _tsp_map = tsp_map;
    const double radius = _tsp_map->get_radius() + SPHERE_RADIUS*2;
    _mat_model = make_scale(Vector3(1.0/radius , 1.0 / radius, 1.0 / radius));
}

void TSPMapDrawer::draw(int code_id /*= 0*/)
{
    draw_route_i();
    draw_map_i();
}

void TSPMapDrawer::set_camera(std::shared_ptr<CameraBase> camera)
{
    _camera = camera;
}

void TSPMapDrawer::set_route(std::vector<int> route)
{
    std::unique_lock<std::mutex> locker(_mutex);
    _route = route;
}

void TSPMapDrawer::draw_map_i()
{
    const double radius = _tsp_map->get_radius();
    const std::vector<Point3>& cities = _tsp_map->get_cities();

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(_camera->get_view_projection_matrix()._m);

    for (int i = 0; i < cities.size(); ++i)
    {
        glColor3d(0.2, 0.2, 0.4);
        if (!_route.empty())
        {
            if (i == _route[0])
            {
                glColor3d(1.0, 0.1, 0.1);
            }
            else if(i ==_route[_route.size()-1])
            {
                glColor3d(0.1, 1.0, 0.1);
            }
        }

        glMatrixMode(GL_MODELVIEW);
        const Matrix4 mat = _mat_model*make_translate(cities[i] - Point3::S_ZERO_POINT);
        glLoadMatrixd(mat._m);

        glutSolidSphere(SPHERE_RADIUS , 20, 20);
    }

    glPopMatrix();
    glPopAttrib();

}

void TSPMapDrawer::draw_route_i()
{
    if (_route.empty())
    {
        return;
    }

    std::vector<Point3> cities;
    {
        std::unique_lock<std::mutex> locker(_mutex);
        cities = _tsp_map->get_cities();
    }

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(_camera->get_view_projection_matrix()._m);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(_mat_model._m);

    
    glColor3d(0.4, 0.2, 0.2);
    for (int i = 0 ; i <_route.size() -1 ; ++i)
    {
        Point3 begin = cities[_route[i]];
        Point3 end = cities[_route[i+1]];

        glLineWidth(0.5 + i*0.015);

        glBegin(GL_LINES);
        glVertex3d(begin.x, begin.y, begin.z);
        glVertex3d(end.x, end.y, end.z);
        glEnd();
    }

    glPopMatrix();
    glPopAttrib();
}
