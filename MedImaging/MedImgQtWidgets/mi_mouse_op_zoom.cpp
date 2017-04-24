#include "mi_mouse_op_zoom.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpZoom::MouseOpZoom()
{

}

MouseOpZoom::~MouseOpZoom()
{

}

void MouseOpZoom::press(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpZoom::move(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }

    _scene->zoom(Point2(_pre_point.x() , _pre_point.y()) , Point2(pt.x() , pt.y()));
    _pre_point = pt;
}

void MouseOpZoom::release(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpZoom::double_click(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpZoom::wheel_slide(int)
{

}

MED_IMAGING_END_NAMESPACE