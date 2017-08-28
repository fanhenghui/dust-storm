#include "mi_mouse_op_pan.h"
#include "renderalgo/mi_scene_base.h"
#include "arithmetic/mi_point2.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpPan::MouseOpPan()
{

}

MouseOpPan::~MouseOpPan()
{

}

void MouseOpPan::press(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpPan::move(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    _scene->pan(Point2(_pre_point.x() , _pre_point.y()) , Point2(pt.x() , pt.y()));
    _pre_point = pt;
}

void MouseOpPan::release(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpPan::double_click(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpPan::wheel_slide(int)
{

}

MED_IMG_END_NAMESPACE