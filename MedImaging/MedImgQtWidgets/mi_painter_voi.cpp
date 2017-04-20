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
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_pScene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_pPainter);

        std::shared_ptr<MPRScene> pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);

        std::shared_ptr<VolumeInfos> pVolumeInfos = pScene->get_volume_infos();
        QTWIDGETS_CHECK_NULL_EXCEPTION(pVolumeInfos);

        int iWidth(1),iHeight(1);
        pScene->get_display_size(iWidth , iHeight);

        //1 Get MPR plane
        std::shared_ptr<CameraBase> pCamera = pScene->get_camera();
        std::shared_ptr<CameraCalculator> pCameraCal = pScene->get_camera_calculator();
        Point3 ptLookAt = pCamera->get_look_at();
        Point3 ptEye = pCamera->get_eye();
        Vector3 vNorm = ptLookAt - ptEye;
        vNorm.normalize();
        Vector3 vUp = pCamera->get_up_direction();

        const Matrix4 matVP = pCamera->get_view_projection_matrix();
        const Matrix4 matP2W = pCameraCal->get_patient_to_world_matrix();

        //2 Calculate sphere intersect with plane
        std::vector<Point2> vecCircleCenter;
        std::vector<int> vecRadius;
        Point3 ptCenter;
        double dDiameter(0.0);
        const std::list<VOISphere> listVOI = m_pModel->get_voi_spheres();
        for (auto it = listVOI.begin() ; it != listVOI.end() ; ++it)
        {
            ptCenter = matP2W.transform(it->m_ptCenter);
            dDiameter = it->m_dDiameter;
            double dDis = vNorm.dot_product(ptLookAt - ptCenter);
            if (abs(dDis) < dDiameter*0.5)
            {
                Point3 pt0 = ptCenter + dDis*vNorm;
                double dRadius = sqrt(dDiameter*dDiameter*0.25 - dDis*dDis);
                Point3 pt1 = pt0 + dRadius*vUp;
                pt0 = matVP.transform(pt0);
                pt1 = matVP.transform(pt1);
                int iSpillTag =0;
                Point2 pt0DC = ArithmeticUtils::ndc_to_dc(Point2(pt0.x , pt0.y) , iWidth , iHeight , iSpillTag);
                Point2 pt1DC = ArithmeticUtils::ndc_to_dc(Point2(pt1.x , pt1.y) , iWidth , iHeight , iSpillTag);
                int iRadius = (int)( (pt1DC - pt0DC).magnitude()+0.5);
                if (iRadius > 1)
                {
                    vecCircleCenter.push_back(pt0DC);
                    vecRadius.push_back(iRadius);
                }
            }
        }

        //3 Draw intersect circle if intersected
        m_pPainter->setPen(QColor(220,50,50));
        for (size_t i = 0 ; i <vecCircleCenter.size() ; ++i)
        {
            m_pPainter->drawEllipse(QPoint((int)vecCircleCenter[i].x , (int)vecCircleCenter[i].y)  , vecRadius[i] , vecRadius[i]);
        }


    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        //assert(false);
        throw e;
    }
}

void VOIPainter::set_voi_model(std::shared_ptr<VOIModel> pModel)
{
    m_pModel = pModel;
}

MED_IMAGING_END_NAMESPACE

