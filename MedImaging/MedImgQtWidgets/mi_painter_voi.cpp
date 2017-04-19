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

void VOIPainter::Render()
{
    try
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_pScene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_pPainter);

        std::shared_ptr<MPRScene> pScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);

        std::shared_ptr<VolumeInfos> pVolumeInfos = pScene->GetVolumeInfos();
        QTWIDGETS_CHECK_NULL_EXCEPTION(pVolumeInfos);

        int iWidth(1),iHeight(1);
        pScene->GetDisplaySize(iWidth , iHeight);

        //1 Get MPR plane
        std::shared_ptr<CameraBase> pCamera = pScene->GetCamera();
        Point3 ptLookAt = pCamera->GetLookAt();
        Point3 ptEye = pCamera->GetEye();
        Vector3 vNorm = ptLookAt - ptEye;
        vNorm.Normalize();
        Vector3 vUp = pCamera->GetUpDirection();

        const Matrix4 matVP = pCamera->GetViewProjectionMatrix();

        //2 Calculate sphere intersect with plane
        std::vector<Point2> vecCircleCenter;
        std::vector<int> vecRadius;
        Point3 ptCenter;
        double dDiameter(0.0);
        const std::list<VOISphere> listVOI = m_pModel->GetVOISpheres();
        for (auto it = listVOI.begin() ; it != listVOI.end() ; ++it)
        {
            ptCenter = it->m_ptCenter;
            dDiameter = it->m_dDiameter;
            double dDis = vNorm.DotProduct(ptLookAt - ptCenter);
            if (abs(dDis) < dDiameter*0.5)
            {
                Point3 pt0 = ptCenter + dDis*vNorm;
                double dRadius = sqrt(dDiameter*dDiameter*0.25 - dDis*dDis);
                Point3 pt1 = pt0 + dRadius*vUp;
                pt0 = matVP.Transform(pt0);
                pt1 = matVP.Transform(pt1);
                int iSpillTag =0;
                Point2 pt0DC = ArithmeticUtils::NDCToDC(Point2(pt0.x , pt0.y) , iWidth , iHeight , iSpillTag);
                Point2 pt1DC = ArithmeticUtils::NDCToDC(Point2(pt1.x , pt1.y) , iWidth , iHeight , iSpillTag);
                int iRadius = (int)( (pt1DC - pt0DC).Magnitude()+0.5);
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

void VOIPainter::SetVOIModel(std::shared_ptr<VOIModel> pModel)
{
    m_pModel = pModel;
}

MED_IMAGING_END_NAMESPACE

