#include "mi_mpr_scene.h"

#include "MedImgCommon/mi_configuration.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"

#include "mi_volume_infos.h"



MED_IMAGING_BEGIN_NAMESPACE

MPRScene::MPRScene():RayCastScene()
{
    std::shared_ptr<MPREntryExitPoints> pMPREE(new MPREntryExitPoints());
    m_pEntryExitPoints = pMPREE;
    if (CPU == Configuration::Instance()->GetProcessingUnitType())
    {
        m_pEntryExitPoints->SetStrategy(CPU_BASE);
    }
    else
    {
        m_pEntryExitPoints->SetStrategy(GPU_BASE);
    }
}

MPRScene::MPRScene(int iWidth , int iHeight):RayCastScene(iWidth , iHeight)
{
    std::shared_ptr<MPREntryExitPoints> pMPREE(new MPREntryExitPoints());
    m_pEntryExitPoints = pMPREE;
    if (CPU == Configuration::Instance()->GetProcessingUnitType())
    {
        m_pEntryExitPoints->SetStrategy(CPU_BASE);
    }
    else
    {
        m_pEntryExitPoints->SetStrategy(GPU_BASE);
    }
}

MPRScene::~MPRScene()
{

}

void MPRScene::PlaceMPR(MedImaging::ScanSliceType eType)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(m_pCameraCalculator);
    //Calculate MPR placement camera
    m_pCameraCalculator->InitMPRPlacement(m_pRayCastCamera , eType);
    //Set initial camera to interactor
    m_pCameraInteractor->SetInitialStatus(m_pRayCastCamera);
    //Resize because initial camera's ratio between width and height  is 1, but current ratio may not.
    m_pCameraInteractor->Resize(m_iWidth , m_iHeight);

    SetDirty(true);
}

void MPRScene::Rotate(const Point2& ptPre , const Point2& ptCur)
{
    m_pCameraInteractor->Rotate(ptPre , ptCur , m_iWidth , m_iHeight );
    SetDirty(true);
}

void MPRScene::Zoom(const Point2& ptPre , const Point2& ptCur)
{
    m_pCameraInteractor->Zoom(ptPre , ptCur , m_iWidth , m_iHeight );
    SetDirty(true);
}

void MPRScene::Pan(const Point2& ptPre , const Point2& ptCur)
{
    m_pCameraInteractor->Pan(ptPre , ptCur , m_iWidth , m_iHeight );
    SetDirty(true);
}

bool MPRScene::GetVolumePosition(const Point2& ptDC , Point3& ptPosV)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolumeInfos);
    std::shared_ptr<ImageData> pImg = m_pVolumeInfos->GetVolume();
    RENDERALGO_CHECK_NULL_EXCEPTION(pImg);

    Point2 pt = ArithmeticUtils::DCToNDC(ptDC , m_iWidth , m_iHeight);

    Matrix4 matMVP = m_pRayCastCamera->GetViewProjectionMatrix()*m_pCameraCalculator->GetVolumeToWorldMatrix();
    matMVP.Inverse();

    Point3 ptVolume = matMVP.Transform(Point3(pt.x , pt.y , 0.0));
    if (ArithmeticUtils::CheckInBound(ptVolume , Point3(pImg->m_uiDim[0] - 1.0 , pImg->m_uiDim[1] - 1 , pImg->m_uiDim[2] - 1)))
    {
        ptPosV = ptVolume;
        return true;
    }
    else
    {
        return false;
    }
}

bool MPRScene::GetWorldPosition(const Point2& ptDC , Point3& ptPosW)
{
    Point3 ptPosV;
    if (GetVolumePosition(ptDC , ptPosV))
    {
        ptPosW = m_pCameraCalculator->GetVolumeToWorldMatrix().Transform(ptPosV);
        return true;
    }
    else
    {
        return false;
    }
}

void MPRScene::Paging(int iStep)
{
    //TODO should consider oblique MPR
    m_pCameraCalculator->MPROrthoPaging(m_pRayCastCamera , iStep);
    SetDirty(true);
}

void MPRScene::PagingTo(int iPage)
{
    m_pCameraCalculator->MPROrthoPagingTo(m_pRayCastCamera , iPage);
    SetDirty(true);
}

Plane MPRScene::ToPlane() const
{
    Point3 ptEye = m_pRayCastCamera->GetEye();
    Point3 ptLookAt = m_pRayCastCamera->GetLookAt();

    Vector3 vNorm = ptLookAt - ptEye;
    vNorm.Normalize();

    Plane p;
    p.m_vNorm = vNorm;
    p.m_dDistance = vNorm.DotProduct(ptLookAt - Point3::kZeroPoint);

    return p;
}

MED_IMAGING_END_NAMESPACE
