#include "mi_graphic_item_cross_hair.h"
#include "mi_model_cross_hair.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

#include <QGraphicsLineItem>
#include <QPen>

MED_IMAGING_BEGIN_NAMESPACE

GraphicItemCrosshair::GraphicItemCrosshair()
{
    _lines[0] = new QGraphicsLineItem();
    _lines[1] = new QGraphicsLineItem();
}

GraphicItemCrosshair::~GraphicItemCrosshair()
{

}

void GraphicItemCrosshair::set_crosshair_model(std::shared_ptr<CrosshairModel> model)
{
    _model = model;
}

std::vector<QGraphicsItem*> GraphicItemCrosshair::get_init_items()
{
    std::vector<QGraphicsItem*> items(2);
    items[0] = _lines[0];
    items[1] = _lines[1];
    return std::move(items);
}

void GraphicItemCrosshair::update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove)
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(_model);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    if (!_model->get_visibility())
    {
        _lines[0]->hide();
        _lines[1]->hide();
        return;
    }
    else
    {
        _lines[0]->show();
        _lines[1]->show();
    }


    Line2D lines[2];
    RGBUnit colors[2];
    _model->get_cross_line(scene , lines , colors);

    if (_pre_lines[0] == lines[0] &&
        _pre_lines[1] == lines[1] &&
        _pre_colors[0] == colors[0] &&
        _pre_colors[1] == colors[1] )
    {
        return;
    }
    else
    {
        _pre_lines[0] == lines[0];
        _pre_lines[1] == lines[1];
        _pre_colors[0] == colors[0];
        _pre_colors[1] == colors[1];
    }

    //std::cout << "<><><><><><><><>\n";
    //std::cout << scene->get_description() << std::endl;

    int width(1) , height(1);
    _scene->get_display_size(width , height);

    for (int i = 0 ; i< 2 ; ++i)
    {
        //Convert to DC
        Line2D line = lines[i];
        line._pt = ArithmeticUtils::ndc_to_dc(lines[i]._pt , width , height);
        line._dir.y = -line._dir.y;

        //Convert to line function a*x + b*y = c
        const double a = -line._dir.y;
        const double b = line._dir.x;
        const double c = a*line._pt.x + b*line._pt.y;

        QPointF pt_dc_0 , pt_dc_1;
        if (abs(a) < DOUBLE_EPSILON)
        {
            double y = c/b;
            y = y < 0 ? 0 : y;
            y = y > (height-1) ? height-1 : y;

            pt_dc_0.setX(0);
            pt_dc_0.setY(y);

            pt_dc_1.setX(width -1);
            pt_dc_1.setY(y);
        }
        else if (abs(b) < DOUBLE_EPSILON)
        {
            double x = c/a;
            x = x < 0 ? 0 : x;
            x = x > (width-1) ? width-1 : x;

            pt_dc_0.setX(x);
            pt_dc_0.setY(0);

            pt_dc_1.setX(x);
            pt_dc_1.setY(height-1);
        }
        else
        {
            //x -1
            double y = (c-a*(-1))/b;
            continue;
        }

        QPen pen(QColor(colors[i].r , colors[i].g , colors[i].b));
        _lines[i]->setPen(pen);
        _lines[i]->setLine(QLineF(pt_dc_0 , pt_dc_1));


    }
    //std::cout << "<><><><><><><><>\n";

}

MED_IMAGING_END_NAMESPACE
