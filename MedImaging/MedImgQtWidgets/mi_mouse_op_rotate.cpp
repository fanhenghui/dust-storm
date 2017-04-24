#include "mi_mouse_op_rotate.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgArithmetic/mi_point2.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpRotate::MouseOpRotate()
{

}

MouseOpRotate::~MouseOpRotate()
{

}

void MouseOpRotate::press(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpRotate::move(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }

    _scene->rotate(Point2(_pre_point.x() , _pre_point.y()) , Point2(pt.x() , pt.y()));
    _pre_point = pt;
}

void MouseOpRotate::release(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpRotate::double_click(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpRotate::wheel_slide(int)
{

}

MED_IMAGING_END_NAMESPACE
