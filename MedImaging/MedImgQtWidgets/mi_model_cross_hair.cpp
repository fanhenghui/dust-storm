#include "mi_model_cross_hair.h"
#include "MedImgArithmetic/mi_intersection_test.h"
#include "MedImgArithmetic/mi_camera_base.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

MED_IMAGING_BEGIN_NAMESPACE

CrosshairModel::CrosshairModel():m_iForceID(0),m_bVisible(true)
{
    m_aPage[0] = 1;
    m_aPage[1] = 1;
    m_aPage[2] = 1;
}

CrosshairModel::~CrosshairModel()
{

}

void CrosshairModel::SetMPRScene(const ScanSliceType (&aScanType)[3] ,const MPRScenePtr (&aMPRScenes)[3] ,const RGBUnit (aMPRColors)[3])
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(aMPRScenes[0]);
    m_pCameraCal = aMPRScenes[0]->GetCameraCalculator();

    for (int i = 0; i<3 ; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(aMPRScenes[i]);
        m_aMPRScene[i] = aMPRScenes[i];
        m_aMPRColor[i] = aMPRColors[i];
        m_aPage[i] =  m_pCameraCal->GetDefaultPage(aScanType[i]);
    }

    m_ptLocationDiscreteW = m_pCameraCal->GetDefaultMPRCenterWorld();
    m_ptLocationContineousW = m_ptLocationDiscreteW;
}

void CrosshairModel::GetCrossLine(const MPRScenePtr& pTargetMPRScene, Line2D (&lines)[2] , RGBUnit (&color)[2])
{
    //1 Choose crossed MPR
    QTWIDGETS_CHECK_NULL_EXCEPTION(pTargetMPRScene);
    MPRScenePtr aCrossScene[2] = {nullptr , nullptr};
    int id = 0;
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            continue;
        }
        aCrossScene[id] = m_aMPRScene[i];
        color[id++] = m_aMPRColor[i];
    }
    assert(id == 2);


    //2 MPR plane intersected to a plane
    const Matrix4 matVP = pTargetMPRScene->GetCamera()->GetViewProjectionMatrix();
    Plane planeTarget = pTargetMPRScene->ToPlane();
    for (int i = 0; i<2; ++i)
    {
        Plane p = aCrossScene[i]->ToPlane();
        Line3D lineIntersect;
        if( IntersectionTest::PlaneIntersectPlane(p, planeTarget,lineIntersect))
        {
            //Project intersected line to screen
            Point3 ptScreen = matVP.Transform(lineIntersect.m_pt);
            lines[i].m_pt = Point2(ptScreen.x , ptScreen.y);
            Vector3 vDir = matVP.GetInverse().GetTranspose().Transform(lineIntersect.m_vDir);
            lines[i].m_vDir = Vector2(vDir.x , vDir.y).GetNormalize();
        }
        else
        {
            lines[i].m_pt = Point2::kZeroPoint;
            lines[i].m_vDir = Vector2(0,0);
        }
    }
}

RGBUnit CrosshairModel::GetBorderColor(MPRScenePtr pTargetMPRScene)
{
    for (int i = 0 ; i< 3 ; ++i)
    {
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            return m_aMPRColor[i];
        }
    }
    return RGBUnit();
}

bool CrosshairModel::PagingTo(const std::shared_ptr<MPRScene>& pTargetMPRScene, int iPage)
{
    //1 Paging target MPR
    int iCurrentPage= GetPage(pTargetMPRScene);
    if (iCurrentPage == iPage)
    {
        return false;
    }

    std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(pTargetMPRScene->GetCamera());
    if( !m_pCameraCal->MPROrthoPagingTo(pCamera , iPage))
    {
        return false;
    }

    pTargetMPRScene->SetDirty(true);
    SetPage_i(pTargetMPRScene , iPage);

    //2 Change cross location
    const Point3 ptCenter = pTargetMPRScene->GetCamera()->GetLookAt();
    const Vector3 vDir = pTargetMPRScene->GetCamera()->GetViewDirection();
    const double dDistance = vDir.DotProduct(ptCenter - m_ptLocationContineousW);
    m_ptLocationContineousW += dDistance*vDir;
    m_ptLocationDiscreteW += dDistance*vDir;

    SetChanged();

    return true;
}

bool CrosshairModel::Paging(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPageStep)
{
    //1 Paging target MPR
    std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(pTargetMPRScene->GetCamera());
    if( !m_pCameraCal->MPROrthoPaging(pCamera , iPageStep))
    {
        return false;
    }

    pTargetMPRScene->SetDirty(true);
    SetPage_i(pTargetMPRScene , m_pCameraCal->GetMPROrthoPage(pCamera));

    //2 Change cross location
    const Point3 ptCenter = pTargetMPRScene->GetCamera()->GetLookAt();
    const Vector3 vDir = pTargetMPRScene->GetCamera()->GetViewDirection();
    const double dDistance = vDir.DotProduct(ptCenter - m_ptLocationContineousW);
    m_ptLocationContineousW += dDistance*vDir;
    m_ptLocationDiscreteW += dDistance*vDir;

    SetChanged();

    return true;
}

bool CrosshairModel::Locate(const std::shared_ptr<MPRScene>& pTargetMPRScene , const Point2& ptDC)
{
    //1 Get latest location
    Point3 ptV;
    if (!pTargetMPRScene->GetVolumePosition(ptDC , ptV))
    {
        return false;
    }

    const Matrix4 matV2W = m_pCameraCal->GetVolumeToWorldMatrix();
    m_ptLocationContineousW = matV2W.Transform(ptV);
    m_ptLocationDiscreteW = matV2W.Transform(Point3( (double)( (int)ptV.x) , (double)( (int)ptV.y) ,(double)( (int)ptV.z) ));

    //2 Choose crossed MPR
    QTWIDGETS_CHECK_NULL_EXCEPTION(pTargetMPRScene);
    MPRScenePtr aCrossScene[2] = {nullptr , nullptr};
    int id = 0;
    int aIdx[2] = {0,0};
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            continue;
        }
        aCrossScene[id] = m_aMPRScene[i];
        aIdx[id++]  = i;
    }
    assert(id == 2);

    //3 Translate crossed MPR( Update LookAt and Update Page)
    for (int i = 0; i<2 ; ++i)
    {
        std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(aCrossScene[i]->GetCamera());
        m_pCameraCal->MPRTranslateTo(pCamera, m_ptLocationContineousW);

        aCrossScene[i]->SetDirty(true);
        int iPage = m_pCameraCal->GetMPROrthoPage(pCamera);
        m_aPage[aIdx[i]] = iPage;
    }

    SetChanged();

    return true;

}

bool CrosshairModel::Locate(const Point3& ptCenterW)
{
    //3 MPR plane paging to the input point slice towards to each normal
    //don't focus the center
    if (!SetCenter_i(ptCenterW))
    {
        return false;
    }

    for (int i = 0 ; i<3 ; ++ i)
    {
        std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(m_aMPRScene[i]->GetCamera());
        m_pCameraCal->MPRTranslateTo(pCamera, m_ptLocationContineousW);

        m_aMPRScene[i]->SetDirty(true);
        int iPage = m_pCameraCal->GetMPROrthoPage(pCamera);
        m_aPage[i] = iPage;
    }

    SetChanged();

    return true;
}

bool CrosshairModel::LocateFocus(const Point3& ptCenterW)
{
    //Place MPR center to this center

    return true;
}

void CrosshairModel::SetPage_i(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPage)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            m_aPage[i] = iPage;
            return;
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Cant find certain MPR scene!");
}

int CrosshairModel::GetPage(const std::shared_ptr<MPRScene>& pTargetMPRScene)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            return m_aPage[i];
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Cant find certain MPR scene!");
}

bool CrosshairModel::SetCenter_i(const Point3& ptCenterW)
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[0]);
    std::shared_ptr<VolumeInfos> pVolumeInfos = m_aMPRScene[0]->GetVolumeInfos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(pVolumeInfos);
    std::shared_ptr<ImageData> pVolume = pVolumeInfos->GetVolume();
    QTWIDGETS_CHECK_NULL_EXCEPTION(pVolume);
    unsigned int *uiDim = pVolume->m_uiDim;

    Point3 ptV = m_pCameraCal->GetWorldToVolumeMatrix().Transform(ptCenterW);
    if (!ArithmeticUtils::CheckInBound(ptV , Point3(uiDim[0] , uiDim[1] , uiDim[2])))
    {
        return false;
    }

    m_ptLocationContineousW = ptCenterW;
    
    m_ptLocationDiscreteW = m_pCameraCal->GetVolumeToWorldMatrix().Transform(
        Point3(double((int)ptV.x) , double((int)ptV.y) , double((int)ptV.z) ));

    return true;
}

bool CrosshairModel::CheckFocus(MPRScenePtr pTargetMPRScene)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            if(m_iForceID == i)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    return false;
}

void CrosshairModel::Focus(MPRScenePtr pTargetMPRScene)
{
    if (!pTargetMPRScene)
    {
        m_iForceID = -1;
    }
    else
    {
        for (int i = 0 ; i< 3; ++i)
        {
            QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
            if (m_aMPRScene[i] == pTargetMPRScene)
            {
                m_iForceID = i;
                break;
            }
        }
    }
}

void CrosshairModel::SetVisibility(bool bFlag)
{
    m_bVisible = bFlag;
}

bool CrosshairModel::GetVisibility() const
{
    return m_bVisible;
}





MED_IMAGING_END_NAMESPACE