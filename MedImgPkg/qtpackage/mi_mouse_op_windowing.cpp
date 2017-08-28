#include "mi_mouse_op_windowing.h"
#include "renderalgo/mi_scene_base.h"
#include "arithmetic/mi_point2.h"
#include "renderalgo/mi_ray_cast_scene.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpWindowing::MouseOpWindowing()
{

}

MouseOpWindowing::~MouseOpWindowing()
{

}

void MouseOpWindowing::press(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpWindowing::move(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }

    //TODO MPR VR diverse strategy
    std::shared_ptr<RayCastScene>  scene = std::dynamic_pointer_cast<RayCastScene>(_scene);
    if (scene)
    {
        float ww , wl;
        scene->get_global_window_level(ww, wl);
        float delta_ww = pt.x() - _pre_point.x();
        float delta_wl = _pre_point.y() - pt.y();
        if (ww + delta_ww > 1.0f)
        {
            ww += delta_ww;
        }
        wl += delta_wl;
        scene->set_global_window_level(ww , wl);
    }

    _pre_point = pt;
}

void MouseOpWindowing::release(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpWindowing::double_click(const QPointF& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpWindowing::wheel_slide(int )
{
}

MED_IMG_END_NAMESPACE