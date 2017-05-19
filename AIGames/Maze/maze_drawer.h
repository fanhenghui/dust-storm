#pragma once

#include <memory>
#include <vector>
#include "Core/matrix4.h"

class Maze;
class CameraBase;
class MazeDrawer
{
public:
    MazeDrawer();
    ~MazeDrawer();

    void set_maze(std::shared_ptr<Maze> maze);

    void set_camera(std::shared_ptr<CameraBase> camera);

    void draw(int code_id = 0);

    //0 north(up)
    //1 south(down)
    //2 east (right)
    //3 west (left)
    void set_route(const std::vector<int>& route);

private:
    void draw_maze_i();
    void draw_route_i();
    void draw_brick_i(int x, int y);// x = cur_column  and y = row - cur_row - 1
    void draw_brick_i(const Point3& begin, const Point3& end);
private:
    std::shared_ptr<Maze> _maze;
    std::vector<int> _route;

    double _brick_size;
    double _margin_size;
    float _wall_color[3];
    float _road_color[3];
    float _entry_color[3];
    float _exit_color[3];

    std::shared_ptr<CameraBase> _camera;
    Matrix4 _mat_model;
};