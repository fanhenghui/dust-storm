#include "mi_painter_voi.h"

#include "MedImgCommon/mi_string_number_converter.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_camera_calculator.h"

#include "mi_model_voi.h"

//Qt
#include <QObject>
#include <QPainter>
#include <QString>
#include <QLabel>

MED_IMAGING_BEGIN_NAMESPACE

VOIPainter::VOIPainter()
{

}

VOIPainter::~VOIPainter()
{

}

void VOIPainter::render()
{
    try
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(_painter);

        std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

        std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
        QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

        int width(1),height(1);
        scene->get_display_size(width , height);

        //1 Get MPR plane
        std::shared_ptr<CameraBase> camera = scene->get_camera();
        std::shared_ptr<CameraCalculator> cameraCal = scene->get_camera_calculator();
        Point3 look_at = camera->get_look_at();
        Point3 eye = camera->get_eye();
        Vector3 norm = look_at - eye;
        norm.normalize();
        Vector3 up = camera->get_up_direction();

        const Matrix4 mat_vp = camera->get_view_projection_matrix();
        const Matrix4 mat_p2w = cameraCal->get_patient_to_world_matrix();

        //2 Calculate sphere intersect with plane
        std::vector<Point2> circle_center;
        std::vector<int> radiuses;
        Point3 sphere_center;
        double diameter(0.0);
        const std::list<VOISphere> voi_list = _model->get_voi_spheres();
        for (auto it = voi_list.begin() ; it != voi_list.end() ; ++it)
        {
            sphere_center = mat_p2w.transform(it->center);
            diameter = it->diameter;
            double distance = norm.dot_product(look_at - sphere_center);
            if (abs(distance) < diameter*0.5)
            {
                Point3 pt0 = sphere_center + distance*norm;
                double radius = sqrt(diameter*diameter*0.25 - distance*distance);
                Point3 pt1 = pt0 + radius*up;
                pt0 = mat_vp.transform(pt0);
                pt1 = mat_vp.transform(pt1);
                int spill_tag =0;
                Point2 pt_dc0 = ArithmeticUtils::ndc_to_dc(Point2(pt0.x , pt0.y) , width , height , spill_tag);
                Point2 pt_dc1 = ArithmeticUtils::ndc_to_dc(Point2(pt1.x , pt1.y) , width , height , spill_tag);
                int radius_int = (int)( (pt_dc1 - pt_dc0).magnitude()+0.5);
                if (radius_int > 1)
                {
                    circle_center.push_back(pt_dc0);
                    radiuses.push_back(radius_int);
                }
            }
        }

        //3 Draw intersect circle if intersected
        _painter->setPen(QColor(220,50,50));
        for (size_t i = 0 ; i <circle_center.size() ; ++i)
        {
            _painter->drawEllipse(QPoint((int)circle_center[i].x , (int)circle_center[i].y)  , radiuses[i] , radiuses[i]);
        }


    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        //assert(false);
        throw e;
    }
}

void VOIPainter::set_voi_model(std::shared_ptr<VOIModel> model)
{
    _model = model;
}

MED_IMAGING_END_NAMESPACE

