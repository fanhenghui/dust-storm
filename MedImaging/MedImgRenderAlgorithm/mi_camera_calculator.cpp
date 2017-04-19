#include "mi_camera_calculator.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

CameraCalculator::CameraCalculator(std::shared_ptr<ImageData> pImgData):m_pImgData(pImgData),
m_bVolumeOrthogonal(true)
{
    Calculate_i();
}
 
//void CameraCalculator::UpdateImageData(std::shared_ptr<ImageData> pImgData)
//{
//    m_pImgData = pImgData;
//    Calculate_i();
//}

const Matrix4& CameraCalculator::GetVolumeToPhysicalMatrix()const
{
    
    return m_matVolume2Physical;
}

const Matrix4& CameraCalculator::GetPhysicalToWorldMatrix()const
{
    return m_matPhysical2World;
}


const Matrix4& CameraCalculator::GetWorldToPatientMatrix()const
{
    return m_matWorld2Patient;
}

const Matrix4& CameraCalculator::GetVolumeToWorldMatrix()const
{
    return m_matVolume2Wolrd;
}

const Matrix4& CameraCalculator::GetWorldToVolumeMatrix()const
{
    return m_matWorld2Volume;
}

void CameraCalculator::InitMPRPlacement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType , const Point3& ptCenterPoint)const
{
    *pCamera = m_OthogonalMPRCamera[(int)eScanSliceType];
    pCamera->SetLookAt(ptCenterPoint);
}

void CameraCalculator::InitMPRPlacement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType)const
{
    *pCamera = m_OthogonalMPRCamera[(int)eScanSliceType];
}

void CameraCalculator::InitVRRPlacement(std::shared_ptr<OrthoCamera> pCamera)const
{
    *pCamera = m_VRPlacementCamera;
}

void CameraCalculator::Calculate_i()
{
    //Check orthogonal 对于CT机架倾斜的数据来说，其扫描的X和Y方向并不正交，渲染需要特殊处理
    CheckVolumeOrthogonal_i();

    //计算体数据的每个轴和标准病人坐标系下的轴的关系
    CalculatePatientAxisInfo_i();

    //Calculate volume to pyhsical/world/patient
    CalculateMatrix_i();

    //Calculate VR replacement
    CalculateVRReplacement_i();

    //Calculate Orthogonal MPR replacement
    CalculateDefaultMPRCenterWorld_i();
    CaluculateOrthogonalMPRReplacement_i();
}

void CameraCalculator::CalculateMatrix_i()
{
    //1 Calculate volume to physical
    unsigned int *uiDim = m_pImgData->m_uiDim;
    double *dSpacing = m_pImgData->m_dSpacing;

    m_matVolume2Physical.SetIdintity();
    m_matVolume2Physical.Prepend(MakeTranslate(-Vector3(uiDim[0] * 0.5, uiDim[1] * 0.5, uiDim[2] * 0.5)));
    m_matVolume2Physical.Prepend(MakeScale(Vector3(dSpacing[0], dSpacing[1], dSpacing[2])));

    //2 Calculate physical to world
    //MatA2B * PA = PB; -----> MatA2B = PB*Inv(PA);
    Vector3 vStanderCoordAxis[6] = {
        Vector3(1.0,0.0,0.0),
        Vector3(-1.0,0.0,0.0),
        Vector3(0.0,1.0,0.0),
        Vector3(0.0,-1.0,0.0),
        Vector3(0.0,0.0,1.0),
        Vector3(0.0,0.0,-1.0)
    };

    const Vector3 vPhysicalHead = vStanderCoordAxis[(int)(m_headInfo.eVolumeCoord)];
    const Vector3 vPhysicalPoterior = vStanderCoordAxis[(int)(m_posteriorInfo.eVolumeCoord)];
    const Vector3 vPhysicalLeft = vStanderCoordAxis[(int)(m_leftInfo.eVolumeCoord)];

    const Matrix4 matPhysical(vPhysicalHead.x,vPhysicalHead.y,vPhysicalHead.z,0.0,
        vPhysicalPoterior.x, vPhysicalPoterior.y, vPhysicalPoterior.z,0.0,
        vPhysicalLeft.x, vPhysicalLeft.y, vPhysicalLeft.z,0.0,
        0.0,0.0,0.0,1.0);

    const Vector3 vPatientHead = m_headInfo.vecInPatient;
    const Vector3 vPatientPosterior = m_posteriorInfo.vecInPatient;
    const Vector3 vPatientLeft = m_leftInfo.vecInPatient;

    const Matrix4 matPatient(vPatientHead.x,vPatientHead.y,vPatientHead.z,0.0,
        vPatientPosterior.x, vPatientPosterior.y, vPatientPosterior.z,0.0,
        vPatientLeft.x, vPatientLeft.y, vPatientLeft.z,0.0,
        0.0,0.0,0.0,1.0);

    if (!matPhysical.HasInverse())
    {
        m_matPhysical2World.SetIdintity();
    }
    else
    {
        m_matPhysical2World = matPatient * matPhysical.GetInverse();
    }

    //3 Calculate volume to world
    m_matVolume2Wolrd = m_matPhysical2World*m_matVolume2Physical;
    m_matWorld2Volume = m_matVolume2Wolrd.GetInverse();

    //4 Calculate world to patient
    const Point3 &ptImgPosition = m_pImgData->m_ptImgPositon;
    const Point3 &ptImgWorld = m_matVolume2Wolrd.Transform(Point3::kZeroPoint);
    m_matWorld2Patient = MakeTranslate(ptImgPosition - ptImgWorld);
}


void CameraCalculator::CalculatePatientAxisInfo_i()
{
    const Vector3& vXCoordPatient = m_pImgData->m_vImgOrientation[0];
    const Vector3& vYCoordPatient = m_pImgData->m_vImgOrientation[1];
    const Vector3& vZCoordPatient = m_pImgData->m_vImgOrientation[2];

    double *dSpacing = m_pImgData->m_dSpacing;
    unsigned int *uiDim = m_pImgData->m_uiDim;

    /// Rotate the volume to get consistent with patient coordinate 
    const Vector3 vStandardHeadAxis(0.0,0.0,1.0);
    const Vector3 vStandardLeftAxis(1.0,0.0,0.0);
    const Vector3 vStandardPosteriorAxis(0.0,1.0,0.0);

    double dRowproduct = vXCoordPatient.DotProduct(vStandardHeadAxis);
    double dColproduct = vYCoordPatient.DotProduct(vStandardHeadAxis);
    double dDepproduct = vZCoordPatient.DotProduct(vStandardHeadAxis);

    /// Priority is TRA>COR>SAG.
    if ((!(abs(dDepproduct) - abs(dRowproduct)<0)) && (!(abs(dDepproduct) - abs(dColproduct)<0)))
    {
        m_headInfo.vecInPatient = dDepproduct > 0? vZCoordPatient : -vZCoordPatient;
        m_headInfo.eVolumeCoord = dDepproduct > 0? POSZ : NEGZ;

        double dLRowproduct = vXCoordPatient.DotProduct(vStandardLeftAxis);
        double dLColproduct = vYCoordPatient.DotProduct(vStandardLeftAxis);

        if (!(abs(dLRowproduct) < abs(dLColproduct)))
        {
            m_leftInfo.vecInPatient = dLRowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_leftInfo.eVolumeCoord = dLRowproduct > 0? POSX : NEGX;

            double dAColproduct = vYCoordPatient.DotProduct(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dAColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_posteriorInfo.eVolumeCoord = dAColproduct > 0? POSY : NEGY;
        }
        else
        {
            m_leftInfo.vecInPatient = dLColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_leftInfo.eVolumeCoord = dLColproduct > 0? POSY : NEGY;

            double dARowproduct = vXCoordPatient.DotProduct(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dARowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_posteriorInfo.eVolumeCoord = dARowproduct > 0? POSX : NEGX;
        }

    }
    else if ((!(abs(dColproduct) - abs(dRowproduct) < 0))&&(!(abs(dColproduct) - abs(dDepproduct) < 0)))
    {
        m_headInfo.vecInPatient = dColproduct > 0? vYCoordPatient : -vYCoordPatient;
        m_headInfo.eVolumeCoord = dColproduct > 0? POSY : NEGY;

        double dLRowproduct  = vXCoordPatient.DotProduct(vStandardLeftAxis);
        double dLDepproduct  = vZCoordPatient.DotProduct(vStandardLeftAxis);

        if (!(abs(dLRowproduct) < abs(dLDepproduct)))
        {
            m_leftInfo.vecInPatient = dLRowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_leftInfo.eVolumeCoord = dLRowproduct > 0? POSX : NEGX;

            double dADepproduct = vZCoordPatient.DotProduct(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dADepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_posteriorInfo.eVolumeCoord = dADepproduct > 0? POSZ : NEGZ;
        }
        else
        {
            m_leftInfo.vecInPatient = dLDepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_leftInfo.eVolumeCoord = dLDepproduct > 0? POSZ : NEGZ;

            double dARowproduct = vXCoordPatient.DotProduct(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dARowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_posteriorInfo.eVolumeCoord = dARowproduct > 0? POSX : NEGX;
        }

    }
    else
    {
        m_headInfo.vecInPatient = dRowproduct > 0? vXCoordPatient : -vXCoordPatient;
        m_headInfo.eVolumeCoord = dRowproduct > 0? POSX : NEGX;

        double dLColproduct  = vYCoordPatient.DotProduct(vStandardLeftAxis);
        double dLDepproduct  = vZCoordPatient.DotProduct(vStandardLeftAxis);

        if (!(abs(dLColproduct) < abs(dLDepproduct)))
        {
            m_leftInfo.vecInPatient = dLColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_leftInfo.eVolumeCoord = dLColproduct > 0? POSY : NEGY;

            double dADepproduct = vZCoordPatient.DotProduct(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dADepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_posteriorInfo.eVolumeCoord = dADepproduct > 0? POSZ : NEGZ;
        }
        else
        {
            m_leftInfo.vecInPatient = dLDepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_leftInfo.eVolumeCoord = dLDepproduct > 0? POSZ : NEGZ;

            double dAColproduct = vYCoordPatient.DotProduct(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dAColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_posteriorInfo.eVolumeCoord = dAColproduct > 0? POSY : NEGY;
        }
    }
}

void CameraCalculator::CalculateVRReplacement_i()
{
    //视角方向和体的平面垂直，即不是和病人坐标系轴平行
    Vector3 vView = Vector3(0.0, 0.0, 0.0);
    
    if (m_bVolumeOrthogonal)
    {
        vView = m_posteriorInfo.vecInPatient;
    }
    else//对于斜扫数据的特殊处理，垂直于面 而非 平行于边
    {
        vView = m_headInfo.vecInPatient.CrossProduct(m_leftInfo.vecInPatient);
    }
    vView.Normalize();

    Vector3 vUp = m_headInfo.vecInPatient;


    double *dSpacing = m_pImgData->m_dSpacing;
    unsigned int *uiDim = m_pImgData->m_uiDim;
    const double dMaxLen = std::max(std::max(uiDim[0] * dSpacing[0],uiDim[1] * dSpacing[1]),uiDim[2] * dSpacing[2]);

    Point3 ptEye = Point3(-vView.x, -vView.y, -vView.z)*dMaxLen*2;
    m_VRPlacementCamera.SetLookAt(Point3(0.0,0.0,0.0));
    m_VRPlacementCamera.SetEye(ptEye);
    m_VRPlacementCamera.SetUpDirection(vUp);
    m_VRPlacementCamera.SetOrtho(-dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*2.0 , dMaxLen*6.0);
}

void CameraCalculator::CheckVolumeOrthogonal_i()
{
    const Vector3& vXCoordPatient = m_pImgData->m_vImgOrientation[0];
    const Vector3& vYCoordPatient = m_pImgData->m_vImgOrientation[1];
    const Vector3& vZCoordPatient = m_pImgData->m_vImgOrientation[2];

    double dXYDot = vXCoordPatient.DotProduct(vYCoordPatient);
    double dXZDot = vXCoordPatient.DotProduct(vZCoordPatient);
    double dYZDot = vYCoordPatient.DotProduct(vZCoordPatient);

    if ( abs(dXYDot) > DOUBLE_EPSILON || abs(dXZDot) > DOUBLE_EPSILON || abs(dYZDot) > DOUBLE_EPSILON )
    {
        m_bVolumeOrthogonal =  false;
    }
    else
    {
        m_bVolumeOrthogonal = true;
    }
}

bool CameraCalculator::GetVolumeOrthogonal()const
{
    return m_bVolumeOrthogonal;
}

const Point3& CameraCalculator::GetDefaultMPRCenterWorld()const
{
    return m_ptDefaultMPRCenter;
}

PatientAxisInfo CameraCalculator::GetHeadPatientAxisInfo()const
{
    return m_headInfo;
}

PatientAxisInfo CameraCalculator::GetLeftPatientAxisInfo()const
{
    return m_leftInfo;
}

PatientAxisInfo CameraCalculator::GetPosteriorPatientAxisInfo()const
{
    return m_posteriorInfo;
}

ScanSliceType CameraCalculator::CheckScaneType(std::shared_ptr<OrthoCamera> pCamera)const
{
    Point3 ptEye = pCamera->GetEye();
    Point3 ptLookAt = pCamera->GetLookAt();
    Vector3 vDir = ptLookAt - ptEye;
    vDir.Normalize();

    const double dDotHead = fabs(m_headInfo.vecInPatient.DotProduct(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.DotProduct(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.DotProduct(vDir));

    ScanSliceType eScanType = OBLIQUE;
    if (dDotHead > dDotLeft && dDotHead > dDotPosterior)//Transverse
    {
        eScanType = fabs(dDotHead - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
    }
    else if (dDotLeft > dDotHead && dDotLeft > dDotPosterior)//Sagittal
    {
        eScanType = fabs(dDotLeft - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
    }
    else//Coronal
    {
        eScanType = fabs(dDotPosterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
    }
    return eScanType;
}

bool CameraCalculator::MPROrthoPaging(std::shared_ptr<OrthoCamera> pCamera , int iPageStep)const
{
    Point3 ptEye = pCamera->GetEye();
    Point3 ptLookAt = pCamera->GetLookAt();
    const Vector3 vDir = pCamera->GetViewDirection();

    const double dDotHead = fabs(m_headInfo.vecInPatient.DotProduct(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.DotProduct(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.DotProduct(vDir));

    double dSpacingStep = 0;
    if (dDotHead > dDotLeft && dDotHead > dDotPosterior)//Transverse
    {
        dSpacingStep = m_pImgData->m_dSpacing[m_headInfo.eVolumeCoord/2];
    }
    else if (dDotLeft > dDotHead && dDotLeft > dDotPosterior)//Sagittal
    {
        dSpacingStep = m_pImgData->m_dSpacing[m_leftInfo.eVolumeCoord/2];
    }
    else//Coronal
    {
        dSpacingStep = m_pImgData->m_dSpacing[m_posteriorInfo.eVolumeCoord/2];
    }

    ptEye += vDir*dSpacingStep*iPageStep;
    ptLookAt+= vDir*dSpacingStep*iPageStep;
    Point3 ptV = m_matWorld2Volume.Transform(ptLookAt);
    if (ArithmeticUtils::CheckInBound(ptV , Point3(m_pImgData->m_uiDim[0]-1 , m_pImgData->m_uiDim[1]-1 , m_pImgData->m_uiDim[2]-1)))
    {
        pCamera->SetEye(ptEye);
        pCamera->SetLookAt(ptLookAt);
        return true;
    }
    else
    {
        return false;
    }
}

bool CameraCalculator::MPROrthoPagingTo(std::shared_ptr<OrthoCamera> pCamera , int iPage)const
{
    //1 Check orthogonal
    const Point3 ptEye = pCamera->GetEye();
    const Point3 ptLookAt = pCamera->GetLookAt();
    const Vector3 vDir = pCamera->GetViewDirection();

    const double dDotHead = fabs(m_headInfo.vecInPatient.DotProduct(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.DotProduct(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.DotProduct(vDir));

    ScanSliceType eScanType = OBLIQUE;
    double dSpacingStep = 0;
    if (dDotHead > dDotLeft && dDotHead > dDotPosterior)//Transverse
    {
        eScanType = fabs(dDotHead - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
        dSpacingStep = m_pImgData->m_dSpacing[m_headInfo.eVolumeCoord/2];
    }
    else if (dDotLeft > dDotHead && dDotLeft > dDotPosterior)//Sagittal
    {
        eScanType = fabs(dDotLeft - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
        dSpacingStep = m_pImgData->m_dSpacing[m_leftInfo.eVolumeCoord/2];
    }
    else//Coronal
    {
        eScanType = fabs(dDotPosterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
        dSpacingStep = m_pImgData->m_dSpacing[m_posteriorInfo.eVolumeCoord/2];
    }

    if (eScanType == OBLIQUE)
    {
        return false;
    }

    //2 Paging
    //2.1 Back to default
    const Point3 ptOriCenter = GetDefaultMPRCenterWorld();
    const Point3 ptOriEye = m_OthogonalMPRCamera[eScanType].GetEye();
    Point3 ptChangedLookAt = ptLookAt + (ptOriCenter - ptLookAt).DotProduct(vDir) * vDir;
    Point3 ptChangedEye = ptEye + (ptOriEye - ptEye).DotProduct(vDir) * vDir;
    //2.2 Paging to certain slice
    const int iStep = iPage - GetDefaultPage(eScanType);
    ptChangedLookAt += vDir* (iStep ) * dSpacingStep;
    ptChangedEye += vDir* (iStep ) * dSpacingStep;

    Point3 ptV = m_matWorld2Volume.Transform(ptChangedLookAt);
    if (ArithmeticUtils::CheckInBound(ptV , Point3(m_pImgData->m_uiDim[0]-1 , m_pImgData->m_uiDim[1]-1 , m_pImgData->m_uiDim[2]-1)))
    {
        pCamera->SetEye(ptChangedEye);
        pCamera->SetLookAt(ptChangedLookAt);
        return true;
    }
    else
    {
        return false;
    }
}

int CameraCalculator::GetMPROrthoPage(std::shared_ptr<OrthoCamera> pCamera) const
{
    //1 Check orthogonal
    const Point3 ptEye = pCamera->GetEye();
    const Point3 ptLookAt = pCamera->GetLookAt();
    const Vector3 vDir = pCamera->GetViewDirection();

    const double dDotHead = fabs(m_headInfo.vecInPatient.DotProduct(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.DotProduct(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.DotProduct(vDir));

    ScanSliceType eScanType = OBLIQUE;
    double dSpacingStep = 0;
    if (dDotHead > dDotLeft && dDotHead > dDotPosterior)//Transverse
    {
        eScanType = fabs(dDotHead - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
        dSpacingStep = m_pImgData->m_dSpacing[m_headInfo.eVolumeCoord/2];
    }
    else if (dDotLeft > dDotHead && dDotLeft > dDotPosterior)//Sagittal
    {
        eScanType = fabs(dDotLeft - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
        dSpacingStep = m_pImgData->m_dSpacing[m_leftInfo.eVolumeCoord/2];
    }
    else//Coronal
    {
        eScanType = fabs(dDotPosterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
        dSpacingStep = m_pImgData->m_dSpacing[m_posteriorInfo.eVolumeCoord/2];
    }

    if (eScanType == OBLIQUE)
    {
        RENDERALGO_THROW_EXCEPTION("Calculate MPR page failed!");
    }

    const double dDistance = vDir.DotProduct(ptLookAt - m_OthogonalMPRCamera[eScanType].GetLookAt());
    int iDelta = int(dDistance/dSpacingStep);
    int iPage = GetDefaultPage(eScanType) + iDelta;
    if (iPage >= 0 && iPage < GetPageMaximum(eScanType))
    {
        return iPage;
    }
    else
    {
        RENDERALGO_THROW_EXCEPTION("Calculate MPR page failed!");
    }

}

void CameraCalculator::MPRTranslateTo(std::shared_ptr<OrthoCamera> pCamera , const Point3& pt)
{
    const Point3 ptLookAt = pCamera->GetLookAt();
    const Point3 ptEye = pCamera->GetEye();
    const Vector3 vDir = pCamera->GetViewDirection();

    const double dTranslate = (pt - ptLookAt).DotProduct(vDir);
    pCamera->SetLookAt(ptLookAt + dTranslate* vDir);
    pCamera->SetEye(ptEye + dTranslate* vDir);
}

float CameraCalculator::ConvertThicknessW2V(std::shared_ptr<OrthoCamera> pMPRCamera , float fThicknessWorldmm)const
{
    Vector3 vDir = (pMPRCamera->GetLookAt() - pMPRCamera->GetEye()).GetNormalize()*fThicknessWorldmm;
    vDir = m_matVolume2Wolrd.GetTranspose().Transform(vDir);
    return (float)(vDir.Magnitude());
}

int CameraCalculator::GetPageMaximum(ScanSliceType eType)const
{
    switch(eType)
    {
    case TRANSVERSE:
        {
            return m_pImgData->m_uiDim[m_headInfo.eVolumeCoord/2];
        }
    case SAGITTAL:
        {
            return m_pImgData->m_uiDim[m_leftInfo.eVolumeCoord/2];
        }
    case CORONAL:
        {
            return m_pImgData->m_uiDim[m_posteriorInfo.eVolumeCoord/2];
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Cant get oblique maximun page!");
        }
    }
}

int CameraCalculator::GetDefaultPage(ScanSliceType eType)const
{
    switch(eType)
    {
    case TRANSVERSE:
        {
            return m_pImgData->m_uiDim[m_headInfo.eVolumeCoord/2]/2;
        }
    case SAGITTAL:
        {
            return m_pImgData->m_uiDim[m_leftInfo.eVolumeCoord/2]/2;
        }
    case CORONAL:
        {
            return m_pImgData->m_uiDim[m_posteriorInfo.eVolumeCoord/2]/2;
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Cant get oblique default page!");
        }
    }
}

void CameraCalculator::CaluculateOrthogonalMPRReplacement_i()
{
    double *dSpacing = m_pImgData->m_dSpacing;
    unsigned int *uiDim = m_pImgData->m_uiDim;
    const double dMaxLen = std::max(std::max(uiDim[0] * dSpacing[0],uiDim[1] * dSpacing[1]),uiDim[2] * dSpacing[2]);

    const Point3 ptLookAt = m_ptDefaultMPRCenter;
    Vector3 vUp;
    Point3 ptEye;
    ScanSliceType aScanType[3] = {SAGITTAL , CORONAL, TRANSVERSE};
    for (int i = 0 ; i< 3 ; ++i)
    {
        switch(aScanType[i])
        {
        case SAGITTAL:
            {
                Vector3 vecx = Vector3(0.0, 0.0, 0.0);
                if (m_bVolumeOrthogonal)
                {
                    vecx = m_leftInfo.vecInPatient;
                }
                else
                {
                    vecx = m_posteriorInfo.vecInPatient.CrossProduct(m_headInfo.vecInPatient);
                }
                vecx.Normalize();

                ptEye = ptLookAt + vecx*dMaxLen*2;
                vUp = Vector3(m_headInfo.vecInPatient);
                break;
            }

        case TRANSVERSE:
            {
                Vector3 vecz = Vector3(0.0, 0.0, 0.0);
                if (m_bVolumeOrthogonal)
                {
                    vecz = -m_headInfo.vecInPatient;
                }
                else
                {
                    vecz = -m_leftInfo.vecInPatient.CrossProduct(m_posteriorInfo.vecInPatient);
                }
                vecz.Normalize();

                ptEye = ptLookAt + vecz*dMaxLen*2;
                vUp = -Vector3(m_posteriorInfo.vecInPatient);
                break;
            }
        case CORONAL:
            {
                Vector3 vecy = Vector3(0.0, 0.0, 0.0);
                if (m_bVolumeOrthogonal)
                {
                    vecy = -m_posteriorInfo.vecInPatient;
                }
                else
                {
                    vecy = -m_headInfo.vecInPatient.CrossProduct(m_leftInfo.vecInPatient);
                }

                vecy.Normalize();

                ptEye = ptLookAt + vecy*dMaxLen*2;
                vUp = Vector3(m_headInfo.vecInPatient);
                break;
            }
        default:
            {
                RENDERALGO_THROW_EXCEPTION("Invalid scan slice type!");
            }
        }

        m_OthogonalMPRCamera[i].SetLookAt(ptLookAt);
        m_OthogonalMPRCamera[i].SetEye(ptEye);
        m_OthogonalMPRCamera[i].SetUpDirection(vUp);
        m_OthogonalMPRCamera[i].SetOrtho(-dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*2.0 , dMaxLen*6.0);
        m_OthogonalMPRNorm[i] = Vector3(ptLookAt - ptEye).GetNormalize();
    }
}

void CameraCalculator::CalculateDefaultMPRCenterWorld_i()
{
    Point3 ptLookAt = Point3(0.0, 0.0, 0.0);
    //Sagittal translate
    Vector3 vViewDir = m_leftInfo.vecInPatient;
    vViewDir.Normalize();
    const unsigned int uiSagittalDimension = m_pImgData->m_uiDim[m_leftInfo.eVolumeCoord/2];
    if ( uiSagittalDimension % 2 == 0 )
    {
        ptLookAt -= vViewDir*0.5*m_pImgData->m_dSpacing[m_leftInfo.eVolumeCoord/2];//取下整
    }

    //Transversal translate
    vViewDir = m_headInfo.vecInPatient;
    vViewDir.Normalize();
    const unsigned int uiTransversalDimension = m_pImgData->m_uiDim[m_headInfo.eVolumeCoord/2];
    if ( 0 == uiTransversalDimension % 2)
    {
        ptLookAt -= vViewDir*0.5*m_pImgData->m_dSpacing[m_headInfo.eVolumeCoord/2];//取下整
    }

    //Coronal translate
    vViewDir = m_posteriorInfo.vecInPatient;
    vViewDir.Normalize();
    const unsigned int uiCoronalDimension = m_pImgData->m_uiDim[m_posteriorInfo.eVolumeCoord/2];
    if ( 0 == uiCoronalDimension % 2)
    {
        ptLookAt -= vViewDir*0.5*m_pImgData->m_dSpacing[m_posteriorInfo.eVolumeCoord/2];//取下整
    }

    m_ptDefaultMPRCenter = ptLookAt;
}

Point3 CameraCalculator::AdjustPointToDiscrete(const Point3& ptWorld) const
{
    Point3 ptVolume = m_matWorld2Volume.Transform(ptWorld);
    ptVolume.x = (double)( (int)ptVolume.x);
    ptVolume.y = (double)( (int)ptVolume.y);
    ptVolume.z = (double)( (int)ptVolume.z);

    return m_matVolume2Wolrd.Transform(ptVolume);
}






MED_IMAGING_END_NAMESPACE