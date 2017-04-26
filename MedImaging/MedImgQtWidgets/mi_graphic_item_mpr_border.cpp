#include "mi_graphic_item_mpr_border.h"

#include "mi_model_cross_hair.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

#include <QGraphicsLineItem>
#include <QPen>

MED_IMAGING_BEGIN_NAMESPACE

GraphicItemMPRBorder::GraphicItemMPRBorder():_pre_color(0,0,0),_pre_pen_width(0)
{
    _lines[0] = new QGraphicsLineItem();
    _lines[1] = new QGraphicsLineItem();
    _lines[2] = new QGraphicsLineItem();
    _lines[3] = new QGraphicsLineItem();
}

GraphicItemMPRBorder::~GraphicItemMPRBorder()
{

}

void GraphicItemMPRBorder::set_crosshair_model(std::shared_ptr<CrosshairModel> model)
{
    _model = model;
}

std::vector<QGraphicsItem*> GraphicItemMPRBorder::get_init_items()
{
    std::vector<QGraphicsItem*> items(4);
    items[0] = _lines[0];
    items[1] = _lines[1];
    items[2] = _lines[2];
    items[3] = _lines[3];
    return std::move(items);
}

void GraphicItemMPRBorder::update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove)
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(_model);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    int width(1) , height(1);
    scene->get_display_size(width , height);

    RGBUnit color = _model->get_border_color(scene);

    QPen pen;
    float border;
    int pen_width;
    QColor color_qt;
    if (_model->check_focus(scene))
    {
        pen_width = 4;
        border = 2.0f;
        color_qt = QColor(0,255,255);
    }
    else
    {
        pen_width = 3;
        border = 1.5f;
        color_qt = QColor(color.r ,color.g , color.b);
    }
    pen.setWidth(pen_width);
    pen.setColor(color_qt);

    if (_pre_color == color_qt && _pre_pen_width == pen_width)
    {
        return;
    }
    else
    {
        _pre_color = color_qt;
        _pre_pen_width= pen_width;
    }


    for (int i = 0; i<4 ; ++i)
    {
        _lines[i]->setPen(pen);
    }

    _lines[0]->setLine(QLineF(QPointF(border,border) , QPointF(border ,height-border)));//left
    _lines[1]->setLine(QLineF(QPointF(width - border,border) , QPointF(width -border ,height-border)));//right
    _lines[2]->setLine(QLineF(QPointF(border,border) , QPointF(width - border ,border)));//top
    _lines[3]->setLine(QLineF(QPointF(border,height - border) , QPointF(width - border ,height - border)));//bottom
}

MED_IMAGING_END_NAMESPACE