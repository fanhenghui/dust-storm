#include "mi_graphic_item_mpr_border.h"
#include "renderalgo/mi_mpr_scene.h"

#include <QGraphicsLineItem>
#include <QPen>

#include "mi_model_cross_hair.h"
#include "mi_model_focus.h"
#include "mi_scene_container.h"

MED_IMG_BEGIN_NAMESPACE 

GraphicItemMPRBorder::GraphicItemMPRBorder():_pre_color(0,0,0),_pre_pen_width(0),_pre_window_width(-1),_pre_window_height(-1)
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
    _model_corsshair = model;
}

void GraphicItemMPRBorder::set_focus_model( std::shared_ptr<FocusModel> model )
{
    _model_focus = model;
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
    QTWIDGETS_CHECK_NULL_EXCEPTION(_model_corsshair);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_model_focus);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    int width(1) , height(1);
    scene->get_display_size(width , height);

    RGBUnit color = _model_corsshair->get_border_color(scene);

    QPen pen;
    float border;
    int pen_width;
    QColor color_qt;

    if (_model_focus->get_focus_scene_container() &&
        _model_focus->get_focus_scene_container()->get_scene() == _scene )
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

    if (_pre_color == color_qt && _pre_pen_width == pen_width &&
        _pre_window_width == width && _pre_window_height== height)
    {
        return;
    }
    else
    {
        _pre_color = color_qt;
        _pre_pen_width= pen_width;
        _pre_window_width = width;
        _pre_window_height = height;
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

MED_IMG_END_NAMESPACE