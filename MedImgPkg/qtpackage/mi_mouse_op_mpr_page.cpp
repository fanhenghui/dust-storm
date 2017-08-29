#include "mi_mouse_op_mpr_page.h"
#include "renderalgo/mi_mpr_scene.h"
#include "arithmetic/mi_point2.h"
#include "mi_model_cross_hair.h"

MED_IMG_BEGIN_NAMESPACE

MouseOpMPRPage::MouseOpMPRPage()
{

}

MouseOpMPRPage::~MouseOpMPRPage()
{

}

void MouseOpMPRPage::press(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpMPRPage::move(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    const int step = int(pt.y() - _pre_point.y());

    if (_model)
    {
        _model->page(scene , step);
        _model->notify();
    }
    else
    {
        scene->page(step);
    }
    _pre_point = pt;
}

void MouseOpMPRPage::release(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpMPRPage::double_click(const QPointF& pt)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpMPRPage::set_crosshair_model(std::shared_ptr<CrosshairModel> model)
{
    _model = model;
}

void MouseOpMPRPage::wheel_slide(int value)
{
    const int step = -value;
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    if (!scene_base)
    {
        return;
    }
    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);
    if (_model)
    {
        _model->page(scene , step);
        _model->notify();
    }
    else
    {
        scene->page(step);
    }
}

MED_IMG_END_NAMESPACE