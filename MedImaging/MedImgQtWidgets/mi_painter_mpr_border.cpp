#include "mi_painter_mpr_border.h"

#include "mi_model_cross_hair.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

MED_IMAGING_BEGIN_NAMESPACE

MPRBorderPainter::MPRBorderPainter()
{

}

MPRBorderPainter::~MPRBorderPainter()
{

}

void MPRBorderPainter::render()
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(_model);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_painter);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    int width(1) , height(1);
    scene->get_display_size(width , height);

    RGBUnit color = _model->get_border_color(scene);

    if (_model->check_focus(scene))
    {
        QPen pen(QColor(0,255,255));
        pen.setWidth(7);
        _painter->setPen(pen);
        _painter->drawRect(QRect(0 , 0 , width, height));
    }
    else
    {
        QPen pen(QColor(color.r ,color.g , color.b));
        pen.setWidth(4);
        _painter->setPen(pen);
        _painter->drawRect(QRect(0 , 0 , width, height));
    }
}

void MPRBorderPainter::set_crosshair_model(std::shared_ptr<CrosshairModel> model)
{
    _model = model;
}

MED_IMAGING_END_NAMESPACE