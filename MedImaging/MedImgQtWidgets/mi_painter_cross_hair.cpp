#include "mi_painter_cross_hair.h"
#include "mi_model_cross_hair.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

MED_IMAGING_BEGIN_NAMESPACE

CrosshairPainter::CrosshairPainter()
{

}

CrosshairPainter::~CrosshairPainter()
{

}

void CrosshairPainter::set_crosshair_model(std::shared_ptr<CrosshairModel> model)
{
    _model = model;
}

void CrosshairPainter::render()
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(_model);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(_painter);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    if (!_model->get_visibility())
    {
        return;
    }


    Line2D lines[2];
    RGBUnit colors[2];
    _model->get_cross_line(scene , lines , colors);

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

        //std::cout << "Pt : ";
        //line._pt.print();
        //std::cout << "  Dir : ";
        //line._dir.print();
        //std::cout << std::endl;

        QPoint pt_dc_0 , pt_dc_1;
        if (abs(a) < DOUBLE_EPSILON)
        {
            double y = c/b;
            int iY = (int)(y+0.5);
            iY = iY < 0 ? 0 : iY;
            iY = iY > (height-1) ? height-1 : iY;

            pt_dc_0.setX(0);
            pt_dc_0.setY(iY);

            pt_dc_1.setX(width -1);
            pt_dc_1.setY(iY);
        }
        else if (abs(b) < DOUBLE_EPSILON)
        {
            double x = c/a;
            int x_int = (int)(x+0.5);
            x_int = x_int < 0 ? 0 : x;
            x_int = x_int > (width-1) ? width-1 : x;

            pt_dc_0.setX(x_int);
            pt_dc_0.setY(0);

            pt_dc_1.setX(x_int);
            pt_dc_1.setY(height-1);
        }
        else
        {
            //x -1
            double y = (c-a*(-1))/b;
            continue;
        }

        _painter->setPen(QColor(colors[i].r , colors[i].g , colors[i].b));
        _painter->drawLine(pt_dc_0 , pt_dc_1);


    }
    //std::cout << "<><><><><><><><>\n";


}

MED_IMAGING_END_NAMESPACE
