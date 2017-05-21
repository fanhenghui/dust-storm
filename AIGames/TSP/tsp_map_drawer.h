#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include "Core/matrix4.h"

class TSPMap;
class CameraBase;
class TSPMapDrawer
{
public:
    TSPMapDrawer();
    ~TSPMapDrawer();

    void set_tps_map(std::shared_ptr<TSPMap> tsp_map);

    void draw(int code_id = 0);

    void set_camera(std::shared_ptr<CameraBase> camera);

    void set_route(std::vector<int> route);

private:
    void draw_map_i();

    void draw_route_i();

private:
    std::shared_ptr<TSPMap> _tsp_map;
    Matrix4 _mat_model;
    std::shared_ptr<CameraBase> _camera;

    std::vector<int> _route;
    std::mutex _mutex;

};