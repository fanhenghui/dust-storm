#include "mi_mouse_op_mpr_paging.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgArithmetic/mi_point2.h"
#include "mi_model_cross_hair.h"

MED_IMAGING_BEGIN_NAMESPACE

MouseOpMPRPaging::MouseOpMPRPaging()
{

}

MouseOpMPRPaging::~MouseOpMPRPaging()
{

}

void MouseOpMPRPaging::press(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }

    _pre_point = pt;
}

void MouseOpMPRPaging::move(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    std::shared_ptr<MPRScene>  scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    const int iStep = int(pt.y() - _pre_point.y());

    if (_model)
    {
        _model->page(scene , iStep);
    }
    else
    {
        scene->page(iStep);
    }
    _pre_point = pt;
}

void MouseOpMPRPaging::release(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpMPRPaging::double_click(const QPoint& pt)
{
    if (!_scene)
    {
        return;
    }
    _pre_point = pt;
}

void MouseOpMPRPaging::set_crosshair_model(std::shared_ptr<CrosshairModel> model)
{
    _model = model;
}

MED_IMAGING_END_NAMESPACE