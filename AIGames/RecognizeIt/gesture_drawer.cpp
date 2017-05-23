#include "gesture_drawer.h"

#include "Ext/gl/glew.h"
#include "Core/matrix4.h"

#define SPLIT_POINT_WIDTH 0.02
GestureDrawer::GestureDrawer(int split):_split_num(split)
{

}

GestureDrawer::~GestureDrawer()
{

}

void GestureDrawer::push_back(const Point2& pt)
{
    _points.push_back(pt);
    const size_t pt_num = _points.size();
    if (pt_num != 1)
    {
        const double cur_l = (_points[pt_num - 1] - _points[pt_num - 2]).magnitude();
            _split_length.push_back(cur_l);
        _route_length += cur_l;
    }

    if (pt_num == 1)
    {
        _split_length.push_back(0);
    }
}

void GestureDrawer::clear()
{
    _points.clear();
    _split_length.clear();
    _split_points.clear();

    _route_length = 0;
}

void GestureDrawer::calculate_splits()
{
    if (_points.size() < 2)
    {
        return;
    }

    _split_points.clear();
    _split_points.reserve(_split_num);
    _split_points.push_back(_points[0]);

    const double ave = _route_length / (_split_num-1);

    for (int i = 0; i < _split_num; ++i)
    {
        double cur_length = 0;
        int j = 0;
        for (; j < _split_length.size(); ++j)
        {
            cur_length += _split_length[j];
            if (cur_length > ave*(i+1))
            {
                break;
            }
        }
        Point2 pt0 = _points[j - 1];
        Point2 pt1 = _points[j ];
        double step = ave*(i + 1) - (cur_length - _split_length[j]);
        Point2 pt = pt0 + (pt1 - pt0).get_normalize()*step;
        _split_points.push_back(pt);
        if (_split_points.size() == _split_num -1)
        {
            break;
        }
    }
    _split_points.push_back(_points[_points.size() - 1]);
}

void GestureDrawer::draw()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //Draw lines 
    if (!_points.empty())
    {
        glColor3d(0.5, 0.5, 0.5);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < _points.size(); ++i)
        {
            glVertex2d(_points[i].x, _points[i].y);
        }
        glEnd();
    }

    //Draw splits
    if (!_split_points.empty())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor3d(0.5, 0.2, 0.5);
        glBegin(GL_QUADS);
        for (int i = 0; i < _split_points.size(); ++i)
        {
            const Point2 &pt = _split_points[i];
            glBegin(GL_QUADS);

            glVertex2d(pt.x-SPLIT_POINT_WIDTH*0.5, pt.y -SPLIT_POINT_WIDTH*0.5);

            glVertex2d(pt.x -SPLIT_POINT_WIDTH*0.5, pt.y + SPLIT_POINT_WIDTH*0.5);

            glVertex2d(pt.x+SPLIT_POINT_WIDTH*0.5, pt.y + SPLIT_POINT_WIDTH*0.5);

            glVertex2d(pt.x+SPLIT_POINT_WIDTH*0.5, pt.y -SPLIT_POINT_WIDTH*0.5);

            glEnd();
        }
        glEnd();
    }

    glPopMatrix();
    glPopAttrib();
}
