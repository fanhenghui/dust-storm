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

void CrosshairPainter::set_crosshair_model(std::shared_ptr<CrosshairModel> pModel)
{
    m_pModel = pModel;
}

void CrosshairPainter::render()
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pModel);
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pScene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_pPainter);

    std::shared_ptr<MPRScene> pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);

    if (!m_pModel->get_visibility())
    {
        return;
    }


    Line2D lines[2];
    RGBUnit colors[2];
    m_pModel->get_cross_line(pScene , lines , colors);

    //std::cout << "<><><><><><><><>\n";
    //std::cout << pScene->get_description() << std::endl;

    int width(1) , height(1);
    m_pScene->get_display_size(width , height);

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

        QPoint pDC0 , pDC1;
        if (abs(a) < DOUBLE_EPSILON)
        {
            double y = c/b;
            int iY = (int)(y+0.5);
            iY = iY < 0 ? 0 : iY;
            iY = iY > (height-1) ? height-1 : iY;

            pDC0.setX(0);
            pDC0.setY(iY);

            pDC1.setX(width -1);
            pDC1.setY(iY);
        }
        else if (abs(b) < DOUBLE_EPSILON)
        {
            double x = c/a;
            int iX = (int)(x+0.5);
            iX = iX < 0 ? 0 : iX;
            iX = iX > (width-1) ? width-1 : iX;

            pDC0.setX(iX);
            pDC0.setY(0);

            pDC1.setX(iX);
            pDC1.setY(height-1);
        }
        else
        {
            //x -1
            double y = (c-a*(-1))/b;
            continue;
        }

        m_pPainter->setPen(QColor(colors[i].r , colors[i].g , colors[i].b));
        m_pPainter->drawLine(pDC0 , pDC1);


    }
    //std::cout << "<><><><><><><><>\n";


}

MED_IMAGING_END_NAMESPACE
