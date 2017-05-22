#include "maze_drawer.h"
#include "maze.h"

#include "Ext/gl/glew.h"
#include "Ext/gl/freeglut.h"
#include "Core/ortho_camera.h"
#include "Core/gl_utils.h"

MazeDrawer::MazeDrawer()
{
    _wall_color[0] = 0.1f;
    _wall_color[1] = 0.1f;
    _wall_color[2] = 0.1f;

    _road_color[0] = 0.9f;
    _road_color[1] = 0.9f;
    _road_color[2] = 0.9f;

    _entry_color[0] = 0.9f;
    _entry_color[1] = 0.1f;
    _entry_color[2] = 0.1f;

    _exit_color[0] = 0.1f;
    _exit_color[1] = 0.9f;
    _exit_color[2] = 0.1f;

    _brick_size = 32.0;
    _margin_size = 1.0;
}

MazeDrawer::~MazeDrawer()
{

}

void MazeDrawer::set_maze(std::shared_ptr<Maze> maze)
{
    _maze = maze;
    int row(0), column(0);
    maze->get_size(row, column);
    double row_w = row*_brick_size + (row + 1)*_margin_size;
    double column_w = column*_brick_size + (column+ 1)*_margin_size;
    const double world_space = std::max(row_w, column_w);
    _mat_model = make_scale(Vector3(2.0/world_space , 2.0 / world_space, 2.0 / world_space))*make_translate(Vector3(-column_w*0.5, -row_w*0.5, -_brick_size*0.5));
}

void MazeDrawer::set_camera(std::shared_ptr<CameraBase> camera)
{
    _camera = camera;
}

void MazeDrawer::draw(int code_id /*= 0*/)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    const Matrix4 mat_mvp = _camera->get_view_projection_matrix()*_mat_model;
    glLoadMatrixd(mat_mvp._m);

    draw_maze_i();

    draw_route_i();

    glPopMatrix();
    glPopAttrib();
}

void MazeDrawer::set_route(const std::vector<int>& route)
{
    _route = route;
}

void MazeDrawer::draw_maze_i()
{
    int* maze_array = _maze->get_maze();
    int row(0), column(0);
    _maze->get_size(row, column);

    glBegin(GL_QUADS);
    float *color = nullptr;
    int unit = 0;
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            unit = maze_array[i*column + j];
            switch (unit)
            {
            case 0:
                color = _road_color;
                break;
            case 1:
                color = _wall_color;
                break;
            case 2:
                color = _entry_color;
                break;
            default://3
                color = _exit_color;
                break;
            }

            if (unit == 0)
            {
                continue;
            }

            int x = j;
            int y = row - i - 1;

            Point3 begin(_margin_size*(x + 1) + x*_brick_size, _margin_size*(y + 1) + y*_brick_size, 0);
            Point3 end = begin + Vector3(_brick_size, _brick_size, _brick_size);

            glColor3f(color[0], color[1], color[2]);
            draw_brick_i(begin, end);
        }
    }
    glEnd();
}

void MazeDrawer::draw_route_i()
{
    if (_route.empty())
    {
        return;
    }

    int* maze_array = _maze->get_maze();
    int row(0), column(0);
    _maze->get_size(row, column);

    int exit_row(0), exit_column(0);
    int entry_row(0), entry_column(0);
    _maze->get_entry(entry_row, entry_column);
    _maze->get_exit(exit_row, exit_column);

    int step = 0;
    int cur_row = entry_row;
    int cur_column = entry_column;
    int cur_unit = 0;

    glColor3d(0.5, 0.9, 0.95);
    glBegin(GL_QUADS);
    for (int i = 0; i < _route.size(); ++i)
    {
        step = _route[i];
        switch (step)
        {
        case NORTH:
        {
            cur_row -= 1;
            if (cur_row < 0)
            {
                cur_row = 0;
            }
            else
            {
                cur_unit = maze_array[cur_row*column + cur_column];
                if (cur_unit == WALL)
                {
                    cur_row += 1;
                }
                else if (cur_unit == EXIT)
                {
                    return;
                }
                else if (cur_unit == ROAD)
                {
                    int x = cur_column;
                    int y = row - cur_row - 1;
                    draw_brick_i(x, y);
                }
            }
            break;
        }
        case SOUTH:
        {
            cur_row += 1;
            if (cur_row > cur_row - 1)
            {
                cur_row = cur_row - 1;
            }
            else
            {
                cur_unit = maze_array[cur_row*column + cur_column];
                if (cur_unit == WALL)
                {
                    cur_row -= 1;
                }
                else if (cur_unit == EXIT)
                {
                    return;
                }
                else if (cur_unit == ROAD)
                {
                    int x = cur_column;
                    int y = row - cur_row - 1;
                    draw_brick_i(x, y);
                }
            }
            break;
        }
        case EAST:
        {
            cur_column += 1;
            if (cur_column > cur_column - 1)
            {
                cur_column = cur_column - 1;
            }
            else
            {
                cur_unit = maze_array[cur_row*column + cur_column];
                if (cur_unit == WALL)
                {
                    cur_column -= 1;
                }
                else if (cur_unit == EXIT)
                {
                    return;
                }
                else if (cur_unit == ROAD)
                {
                    int x = cur_column;
                    int y = row - cur_row - 1;
                    draw_brick_i(x, y);
                }
            }
            break;
        }
        default://WEST
        {
            cur_column -= 1;
            if (cur_column < 0)
            {
                cur_column = 0;
            }
            else
            {
                cur_unit = maze_array[cur_row*column + cur_column];
                if (cur_unit == WALL)
                {
                    cur_column += 1;
                }
                else if (cur_unit == EXIT)
                {
                    return;
                }
                else if (cur_unit == ROAD)
                {
                    int x = cur_column;
                    int y = row - cur_row - 1;
                    draw_brick_i(x, y);
                }
            }
            break;
        }
        }
    }

    glEnd();
}

void MazeDrawer::draw_brick_i(const Point3& begin, const Point3& end)
{
#define PER_VERTEX( x ,  y ,  z)  glVertex3d(x, y , z);

    PER_VERTEX(begin.x, begin.y, begin.z);
    PER_VERTEX(end.x, begin.y, begin.z);
    PER_VERTEX(end.x, end.y, begin.z);
    PER_VERTEX(begin.x, end.y, begin.z);

    //Z +
    PER_VERTEX(begin.x, begin.y, end.z);
    PER_VERTEX(end.x, begin.y, end.z);
    PER_VERTEX(end.x, end.y, end.z);
    PER_VERTEX(begin.x, end.y, end.z);

    //X-
    PER_VERTEX(begin.x, begin.y, begin.z);
    PER_VERTEX(begin.x, end.y, begin.z);
    PER_VERTEX(begin.x, end.y, end.z);
    PER_VERTEX(begin.x, begin.y, end.z);

    //X+
    PER_VERTEX(end.x, begin.y, begin.z);
    PER_VERTEX(end.x, end.y, begin.z);
    PER_VERTEX(end.x, end.y, end.z);
    PER_VERTEX(end.x, begin.y, end.z);

    //Y-
    PER_VERTEX(begin.x, begin.y, begin.z);
    PER_VERTEX(begin.x, begin.y, end.z);
    PER_VERTEX(end.x, begin.y, end.z);
    PER_VERTEX(end.x, begin.y, begin.z);

    //Y+
    PER_VERTEX(begin.x, end.y, begin.z);
    PER_VERTEX(begin.x, end.y, end.z);
    PER_VERTEX(end.x, end.y, end.z);
    PER_VERTEX(end.x, end.y, begin.z);

#undef PER_VERTEX
}

void MazeDrawer::draw_brick_i(int x, int y)
{
    Point3 begin(_margin_size*(x + 1) + x*_brick_size, _margin_size*(y + 1) + y*_brick_size, 0);
    Point3 end = begin + Vector3(_brick_size, _brick_size, _brick_size);

    draw_brick_i(begin, end);
}

