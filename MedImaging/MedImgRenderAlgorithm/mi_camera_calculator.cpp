#include "mi_camera_calculator.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

CameraCalculator::CameraCalculator(std::shared_ptr<ImageData> pImgData):m_pImgData(pImgData),
m_bVolumeOrthogonal(true)
{
    calculate_i();
}
 
//void CameraCalculator::UpdateImageData(std::shared_ptr<ImageData> pImgData)
//{
//    m_pImgData = pImgData;
//    Calculate_i();
//}

const Matrix4& CameraCalculator::get_volume_to_physical_matrix()const
{
    
    return m_matVolume2Physical;
}

const Matrix4& CameraCalculator::get_physical_to_world_matrix()const
{
    return m_matPhysical2World;
}


const Matrix4& CameraCalculator::get_world_to_patient_matrix()const
{
    return m_matWorld2Patient;
}

const Matrix4& CameraCalculator::get_volume_to_world_matrix()const
{
    return m_matVolume2Wolrd;
}

const Matrix4& CameraCalculator::get_world_to_volume_matrix()const
{
    return m_matWorld2Volume;
}

void CameraCalculator::init_mpr_placement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType , const Point3& ptCenterPoint)const
{
    *pCamera = m_OthogonalMPRCamera[(int)eScanSliceType];
    pCamera->set_look_at(ptCenterPoint);
}

void CameraCalculator::init_mpr_placement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType)const
{
    *pCamera = m_OthogonalMPRCamera[(int)eScanSliceType];
}

void CameraCalculator::init_vr_placement(std::shared_ptr<OrthoCamera> pCamera)const
{
    *pCamera = m_VRPlacementCamera;
}

void CameraCalculator::calculate_i()
{
    //Check orthogonal 对于CT机架倾斜的数据来说，其扫描的X和Y方向并不正交，渲染需要特殊处理
    check_volume_orthogonal_i();

    //计算体数据的每个轴和标准病人坐标系下的轴的关系
    calculate_patient_axis_info_i();

    //Calculate volume to pyhsical/world/patient
    calculate_matrix_i();

    //Calculate VR replacement
    calculate_vr_placement_i();

    //Calculate orthogonal MPR replacement
    calculate_default_mpr_center_world_i();
    caluculate_orthogonal_mpr_placement_i();
}

void CameraCalculator::calculate_matrix_i()
{
    //1 Calculate volume to physical
    unsigned int *uiDim = m_pImgData->m_uiDim;
    double *dSpacing = m_pImgData->m_dSpacing;

    m_matVolume2Physical.set_idintity();
    m_matVolume2Physical.prepend(make_translate(-Vector3(uiDim[0] * 0.5, uiDim[1] * 0.5, uiDim[2] * 0.5)));
    m_matVolume2Physical.prepend(make_scale(Vector3(dSpacing[0], dSpacing[1], dSpacing[2])));

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

    if (!matPhysical.has_inverse())
    {
        m_matPhysical2World.set_idintity();
    }
    else
    {
        m_matPhysical2World = matPatient * matPhysical.get_inverse();
    }

    //3 Calculate volume to world
    m_matVolume2Wolrd = m_matPhysical2World*m_matVolume2Physical;
    m_matWorld2Volume = m_matVolume2Wolrd.get_inverse();

    //4 Calculate world to patient
    const Point3 &ptImgPosition = m_pImgData->m_ptImgPositon;
    const Point3 &ptImgWorld = m_matVolume2Wolrd.transform(Point3::kZeroPoint);
    m_matWorld2Patient = make_translate(ptImgPosition - ptImgWorld);
}


void CameraCalculator::calculate_patient_axis_info_i()
{
    const Vector3& vXCoordPatient = m_pImgData->m_vImgOrientation[0];
    const Vector3& vYCoordPatient = m_pImgData->m_vImgOrientation[1];
    const Vector3& vZCoordPatient = m_pImgData->m_vImgOrientation[2];

    double *dSpacing = m_pImgData->m_dSpacing;
    unsigned int *uiDim = m_pImgData->m_uiDim;

    /// rotate the volume to get consistent with patient coordinate 
    const Vector3 vStandardHeadAxis(0.0,0.0,1.0);
    const Vector3 vStandardLeftAxis(1.0,0.0,0.0);
    const Vector3 vStandardPosteriorAxis(0.0,1.0,0.0);

    double dRowproduct = vXCoordPatient.dot_product(vStandardHeadAxis);
    double dColproduct = vYCoordPatient.dot_product(vStandardHeadAxis);
    double dDepproduct = vZCoordPatient.dot_product(vStandardHeadAxis);

    /// Priority is TRA>COR>SAG.
    if ((!(abs(dDepproduct) - abs(dRowproduct)<0)) && (!(abs(dDepproduct) - abs(dColproduct)<0)))
    {
        m_headInfo.vecInPatient = dDepproduct > 0? vZCoordPatient : -vZCoordPatient;
        m_headInfo.eVolumeCoord = dDepproduct > 0? POSZ : NEGZ;

        double dLRowproduct = vXCoordPatient.dot_product(vStandardLeftAxis);
        double dLColproduct = vYCoordPatient.dot_product(vStandardLeftAxis);

        if (!(abs(dLRowproduct) < abs(dLColproduct)))
        {
            m_leftInfo.vecInPatient = dLRowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_leftInfo.eVolumeCoord = dLRowproduct > 0? POSX : NEGX;

            double dAColproduct = vYCoordPatient.dot_product(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dAColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_posteriorInfo.eVolumeCoord = dAColproduct > 0? POSY : NEGY;
        }
        else
        {
            m_leftInfo.vecInPatient = dLColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_leftInfo.eVolumeCoord = dLColproduct > 0? POSY : NEGY;

            double dARowproduct = vXCoordPatient.dot_product(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dARowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_posteriorInfo.eVolumeCoord = dARowproduct > 0? POSX : NEGX;
        }

    }
    else if ((!(abs(dColproduct) - abs(dRowproduct) < 0))&&(!(abs(dColproduct) - abs(dDepproduct) < 0)))
    {
        m_headInfo.vecInPatient = dColproduct > 0? vYCoordPatient : -vYCoordPatient;
        m_headInfo.eVolumeCoord = dColproduct > 0? POSY : NEGY;

        double dLRowproduct  = vXCoordPatient.dot_product(vStandardLeftAxis);
        double dLDepproduct  = vZCoordPatient.dot_product(vStandardLeftAxis);

        if (!(abs(dLRowproduct) < abs(dLDepproduct)))
        {
            m_leftInfo.vecInPatient = dLRowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_leftInfo.eVolumeCoord = dLRowproduct > 0? POSX : NEGX;

            double dADepproduct = vZCoordPatient.dot_product(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dADepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_posteriorInfo.eVolumeCoord = dADepproduct > 0? POSZ : NEGZ;
        }
        else
        {
            m_leftInfo.vecInPatient = dLDepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_leftInfo.eVolumeCoord = dLDepproduct > 0? POSZ : NEGZ;

            double dARowproduct = vXCoordPatient.dot_product(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dARowproduct > 0? vXCoordPatient : -vXCoordPatient;
            m_posteriorInfo.eVolumeCoord = dARowproduct > 0? POSX : NEGX;
        }

    }
    else
    {
        m_headInfo.vecInPatient = dRowproduct > 0? vXCoordPatient : -vXCoordPatient;
        m_headInfo.eVolumeCoord = dRowproduct > 0? POSX : NEGX;

        double dLColproduct  = vYCoordPatient.dot_product(vStandardLeftAxis);
        double dLDepproduct  = vZCoordPatient.dot_product(vStandardLeftAxis);

        if (!(abs(dLColproduct) < abs(dLDepproduct)))
        {
            m_leftInfo.vecInPatient = dLColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_leftInfo.eVolumeCoord = dLColproduct > 0? POSY : NEGY;

            double dADepproduct = vZCoordPatient.dot_product(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dADepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_posteriorInfo.eVolumeCoord = dADepproduct > 0? POSZ : NEGZ;
        }
        else
        {
            m_leftInfo.vecInPatient = dLDepproduct > 0? vZCoordPatient : -vZCoordPatient;
            m_leftInfo.eVolumeCoord = dLDepproduct > 0? POSZ : NEGZ;

            double dAColproduct = vYCoordPatient.dot_product(vStandardPosteriorAxis);
            m_posteriorInfo.vecInPatient = dAColproduct > 0? vYCoordPatient : -vYCoordPatient;
            m_posteriorInfo.eVolumeCoord = dAColproduct > 0? POSY : NEGY;
        }
    }
}

void CameraCalculator::calculate_vr_placement_i()
{
    //视角方向和体的平面垂直，即不是和病人坐标系轴平行
    Vector3 vView = Vector3(0.0, 0.0, 0.0);
    
    if (m_bVolumeOrthogonal)
    {
        vView = m_posteriorInfo.vecInPatient;
    }
    else//对于斜扫数据的特殊处理，垂直于面 而非 平行于边
    {
        vView = m_headInfo.vecInPatient.cross_product(m_leftInfo.vecInPatient);
    }
    vView.normalize();

    Vector3 vUp = m_headInfo.vecInPatient;


    double *dSpacing = m_pImgData->m_dSpacing;
    unsigned int *uiDim = m_pImgData->m_uiDim;
    const double dMaxLen = std::max(std::max(uiDim[0] * dSpacing[0],uiDim[1] * dSpacing[1]),uiDim[2] * dSpacing[2]);

    Point3 ptEye = Point3(-vView.x, -vView.y, -vView.z)*dMaxLen*2;
    m_VRPlacementCamera.set_look_at(Point3(0.0,0.0,0.0));
    m_VRPlacementCamera.set_eye(ptEye);
    m_VRPlacementCamera.set_up_direction(vUp);
    m_VRPlacementCamera.set_ortho(-dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*2.0 , dMaxLen*6.0);
}

void CameraCalculator::check_volume_orthogonal_i()
{
    const Vector3& vXCoordPatient = m_pImgData->m_vImgOrientation[0];
    const Vector3& vYCoordPatient = m_pImgData->m_vImgOrientation[1];
    const Vector3& vZCoordPatient = m_pImgData->m_vImgOrientation[2];

    double dXYDot = vXCoordPatient.dot_product(vYCoordPatient);
    double dXZDot = vXCoordPatient.dot_product(vZCoordPatient);
    double dYZDot = vYCoordPatient.dot_product(vZCoordPatient);

    if ( abs(dXYDot) > DOUBLE_EPSILON || abs(dXZDot) > DOUBLE_EPSILON || abs(dYZDot) > DOUBLE_EPSILON )
    {
        m_bVolumeOrthogonal =  false;
    }
    else
    {
        m_bVolumeOrthogonal = true;
    }
}

bool CameraCalculator::check_volume_orthogonal()const
{
    return m_bVolumeOrthogonal;
}

const Point3& CameraCalculator::get_default_mpr_center_world()const
{
    return m_ptDefaultMPRCenter;
}

PatientAxisInfo CameraCalculator::get_head_patient_axis_info()const
{
    return m_headInfo;
}

PatientAxisInfo CameraCalculator::get_left_patient_axis_info()const
{
    return m_leftInfo;
}

PatientAxisInfo CameraCalculator::get_posterior_patient_axis_info()const
{
    return m_posteriorInfo;
}

ScanSliceType CameraCalculator::check_scan_type(std::shared_ptr<OrthoCamera> pCamera)const
{
    Point3 ptEye = pCamera->get_eye();
    Point3 ptLookAt = pCamera->get_look_at();
    Vector3 vDir = ptLookAt - ptEye;
    vDir.normalize();

    const double dDotHead = fabs(m_headInfo.vecInPatient.dot_product(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.dot_product(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.dot_product(vDir));

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

bool CameraCalculator::page_orthognal_mpr(std::shared_ptr<OrthoCamera> pCamera , int iPageStep)const
{
    Point3 ptEye = pCamera->get_eye();
    Point3 ptLookAt = pCamera->get_look_at();
    const Vector3 vDir = pCamera->get_view_direction();

    const double dDotHead = fabs(m_headInfo.vecInPatient.dot_product(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.dot_product(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.dot_product(vDir));

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
    Point3 ptV = m_matWorld2Volume.transform(ptLookAt);
    if (ArithmeticUtils::check_in_bound(ptV , Point3(m_pImgData->m_uiDim[0]-1 , m_pImgData->m_uiDim[1]-1 , m_pImgData->m_uiDim[2]-1)))
    {
        pCamera->set_eye(ptEye);
        pCamera->set_look_at(ptLookAt);
        return true;
    }
    else
    {
        return false;
    }
}

bool CameraCalculator::page_orthognal_mpr_to(std::shared_ptr<OrthoCamera> pCamera , int iPage)const
{
    //1 Check orthogonal
    const Point3 ptEye = pCamera->get_eye();
    const Point3 ptLookAt = pCamera->get_look_at();
    const Vector3 vDir = pCamera->get_view_direction();

    const double dDotHead = fabs(m_headInfo.vecInPatient.dot_product(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.dot_product(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.dot_product(vDir));

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

    //2 page
    //2.1 Back to default
    const Point3 ptOriCenter = get_default_mpr_center_world();
    const Point3 ptOriEye = m_OthogonalMPRCamera[eScanType].get_eye();
    Point3 ptChangedLookAt = ptLookAt + (ptOriCenter - ptLookAt).dot_product(vDir) * vDir;
    Point3 ptChangedEye = ptEye + (ptOriEye - ptEye).dot_product(vDir) * vDir;
    //2.2 page to certain slice
    const int iStep = iPage - get_default_page(eScanType);
    ptChangedLookAt += vDir* (iStep ) * dSpacingStep;
    ptChangedEye += vDir* (iStep ) * dSpacingStep;

    Point3 ptV = m_matWorld2Volume.transform(ptChangedLookAt);
    if (ArithmeticUtils::check_in_bound(ptV , Point3(m_pImgData->m_uiDim[0]-1 , m_pImgData->m_uiDim[1]-1 , m_pImgData->m_uiDim[2]-1)))
    {
        pCamera->set_eye(ptChangedEye);
        pCamera->set_look_at(ptChangedLookAt);
        return true;
    }
    else
    {
        return false;
    }
}

int CameraCalculator::get_orthognal_mpr_page(std::shared_ptr<OrthoCamera> pCamera) const
{
    //1 Check orthogonal
    const Point3 ptEye = pCamera->get_eye();
    const Point3 ptLookAt = pCamera->get_look_at();
    const Vector3 vDir = pCamera->get_view_direction();

    const double dDotHead = fabs(m_headInfo.vecInPatient.dot_product(vDir));
    const double dDotLeft = fabs(m_leftInfo.vecInPatient.dot_product(vDir));
    const double dDotPosterior = fabs(m_posteriorInfo.vecInPatient.dot_product(vDir));

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

    const double dDistance = vDir.dot_product(ptLookAt - m_OthogonalMPRCamera[eScanType].get_look_at());
    int iDelta = int(dDistance/dSpacingStep);
    int iPage = get_default_page(eScanType) + iDelta;
    if (iPage >= 0 && iPage < get_page_maximum(eScanType))
    {
        return iPage;
    }
    else
    {
        RENDERALGO_THROW_EXCEPTION("Calculate MPR page failed!");
    }

}

void CameraCalculator::translate_mpr_to(std::shared_ptr<OrthoCamera> pCamera , const Point3& pt)
{
    const Point3 ptLookAt = pCamera->get_look_at();
    const Point3 ptEye = pCamera->get_eye();
    const Vector3 vDir = pCamera->get_view_direction();

    const double dTranslate = (pt - ptLookAt).dot_product(vDir);
    pCamera->set_look_at(ptLookAt + dTranslate* vDir);
    pCamera->set_eye(ptEye + dTranslate* vDir);
}

//float CameraCalculator::convert_thickness_world_to_volume(std::shared_ptr<OrthoCamera> pMPRCamera , float fThicknessWorldmm)const
//{
//    Vector3 vDir = (pMPRCamera->get_look_at() - pMPRCamera->get_eye()).get_normalize()*fThicknessWorldmm;
//    vDir = m_matVolume2Wolrd.get_transpose().transform(vDir);
//    return (float)(vDir.magnitude());
//}

int CameraCalculator::get_page_maximum(ScanSliceType eType)const
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

int CameraCalculator::get_default_page(ScanSliceType eType)const
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

void CameraCalculator::caluculate_orthogonal_mpr_placement_i()
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
                    vecx = m_posteriorInfo.vecInPatient.cross_product(m_headInfo.vecInPatient);
                }
                vecx.normalize();

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
                    vecz = -m_leftInfo.vecInPatient.cross_product(m_posteriorInfo.vecInPatient);
                }
                vecz.normalize();

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
                    vecy = -m_headInfo.vecInPatient.cross_product(m_leftInfo.vecInPatient);
                }

                vecy.normalize();

                ptEye = ptLookAt + vecy*dMaxLen*2;
                vUp = Vector3(m_headInfo.vecInPatient);
                break;
            }
        default:
            {
                RENDERALGO_THROW_EXCEPTION("Invalid scan slice type!");
            }
        }

        m_OthogonalMPRCamera[i].set_look_at(ptLookAt);
        m_OthogonalMPRCamera[i].set_eye(ptEye);
        m_OthogonalMPRCamera[i].set_up_direction(vUp);
        m_OthogonalMPRCamera[i].set_ortho(-dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*0.5 , dMaxLen*0.5 , -dMaxLen*2.0 , dMaxLen*6.0);
        m_OthogonalMPRNorm[i] = Vector3(ptLookAt - ptEye).get_normalize();
    }
}

void CameraCalculator::calculate_default_mpr_center_world_i()
{
    Point3 ptLookAt = Point3(0.0, 0.0, 0.0);
    //Sagittal translate
    Vector3 vViewDir = m_leftInfo.vecInPatient;
    vViewDir.normalize();
    const unsigned int uiSagittalDimension = m_pImgData->m_uiDim[m_leftInfo.eVolumeCoord/2];
    if ( uiSagittalDimension % 2 == 0 )
    {
        ptLookAt -= vViewDir*0.5*m_pImgData->m_dSpacing[m_leftInfo.eVolumeCoord/2];//取下整
    }

    //Transversal translate
    vViewDir = m_headInfo.vecInPatient;
    vViewDir.normalize();
    const unsigned int uiTransversalDimension = m_pImgData->m_uiDim[m_headInfo.eVolumeCoord/2];
    if ( 0 == uiTransversalDimension % 2)
    {
        ptLookAt -= vViewDir*0.5*m_pImgData->m_dSpacing[m_headInfo.eVolumeCoord/2];//取下整
    }

    //Coronal translate
    vViewDir = m_posteriorInfo.vecInPatient;
    vViewDir.normalize();
    const unsigned int uiCoronalDimension = m_pImgData->m_uiDim[m_posteriorInfo.eVolumeCoord/2];
    if ( 0 == uiCoronalDimension % 2)
    {
        ptLookAt -= vViewDir*0.5*m_pImgData->m_dSpacing[m_posteriorInfo.eVolumeCoord/2];//取下整
    }

    m_ptDefaultMPRCenter = ptLookAt;
}

//Point3 CameraCalculator::adjust_point_to_discrete(const Point3& ptWorld) const
//{
//    Point3 ptVolume = m_matWorld2Volume.transform(ptWorld);
//    ptVolume.x = (double)( (int)ptVolume.x);
//    ptVolume.y = (double)( (int)ptVolume.y);
//    ptVolume.z = (double)( (int)ptVolume.z);
//
//    return m_matVolume2Wolrd.transform(ptVolume);
//}






MED_IMAGING_END_NAMESPACE