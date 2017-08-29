#include "mi_mouse_op_rotate.h"
#include "renderalgo/mi_scene_base.h"
#include "arithmetic/mi_point2.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpRotate::MouseOpRotate()
{

}

MouseOpRotate::~MouseOpRotate()
{

}

void MouseOpRotate::press(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpRotate::move(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    scene_base->rotate(Point2(_pre_point.x() , _pre_point.y()) , Point2(pt.x() , pt.y()));
    _pre_point = pt;
}

void MouseOpRotate::release(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpRotate::double_click(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpRotate::wheel_slide(int)
{

}

MED_IMG_END_NAMESPACE
