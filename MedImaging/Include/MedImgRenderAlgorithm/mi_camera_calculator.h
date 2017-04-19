#ifndef MI_MED_IMAGING_CAMERA_CALCULATOR_H_
#define MI_MED_IMAGING_CAMERA_CALCULATOR_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3.h"
#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_ortho_camera.h"

MED_IMAGING_BEGIN_NAMESPACE

class ImageData;
class CameraBase;
class OrthoCamera;

//volume坐标系坐标轴
enum CoordinateAxis
{
    POSX = 0,
    NEGX = 1,
    POSY = 2,
    NEGY = 3,
    POSZ = 4,
    NEGZ = 5,
};

//病人方位信息
struct PatientAxisInfo
{
    //与该patient坐标系 最靠近 的 volume坐标系轴方向
    //EG：PatientAxisInfo leftAxisInfo; leftAxisInfo.eVolumeCoord = POSX 说明病人坐标系Left方向最近的体坐标系是正X方向
    CoordinateAxis eVolumeCoord;

    //在patient坐标系 最靠近 的 volume坐标系轴方向 的实际patient方向
    //EG : 同上，leftAxisInfo.vecPatient = (0.9 0 0) , 说明和标准的 (1,0,0) 有偏移
    // 这是为了在world 坐标系下（方向和病人坐标系相同）能够调整相机的位置（未必和病人坐标系轴平行）来观察volume的正方向（为了看扫描位），而非patient坐标的正方向
    Vector3 vecInPatient;
};

enum ScanSliceType
{
    SAGITTAL= 0,
    CORONAL = 1,
    TRANSVERSE = 2,
    OBLIQUE = 3,
};

class RenderAlgo_Export CameraCalculator
{
public:
    CameraCalculator(std::shared_ptr<ImageData> pImgData);

    //void UpdateImageData(std::shared_ptr<ImageData> pImgData);

    //////////////////////////////////////////////////////////////////////////
    //Initializion 
    //////////////////////////////////////////////////////////////////////////

    //仅仅改变了度量，从pixel 到 mm ，且在这个坐标系下的Volume bounding box仍然是axis aligned
    const Matrix4& GetVolumeToPhysicalMatrix() const;

    //旋转到了和病人坐标系轴对齐，世界坐标系的原点是体数据体素中心点
    const Matrix4& GetPhysicalToWorldMatrix() const;

    const Matrix4& GetVolumeToWorldMatrix() const;

    const Matrix4& GetWorldToVolumeMatrix() const;

    //病人坐标系和世界坐标系仅仅是坐标原点不一样而已
    const Matrix4& GetWorldToPatientMatrix() const;

    const Point3& GetDefaultMPRCenterWorld() const;

    void InitMPRPlacement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType , const Point3& ptWorldCenterPoint) const;//实时获取

    void InitMPRPlacement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType ) const;//Default replacement

    void InitVRRPlacement(std::shared_ptr<OrthoCamera> pCamera) const;

    PatientAxisInfo GetHeadPatientAxisInfo() const;

    PatientAxisInfo GetLeftPatientAxisInfo() const;

    PatientAxisInfo GetPosteriorPatientAxisInfo() const;

    int GetPageMaximum(ScanSliceType eType) const;

    int GetDefaultPage(ScanSliceType eType) const;

    bool GetVolumeOrthogonal() const;

    //////////////////////////////////////////////////////////////////////////
    //Interaction
    //////////////////////////////////////////////////////////////////////////

    //Default positive direction
    // Axial F->H
    // Sagittal R->L
    // Coronal A -> P
    bool MPROrthoPaging(std::shared_ptr<OrthoCamera> pMPRCamera , int iPageStep) const;

    bool MPROrthoPagingTo(std::shared_ptr<OrthoCamera> pMPRCamera , int iPage) const;

    int GetMPROrthoPage(std::shared_ptr<OrthoCamera> pMPRCamera) const;

    //MPR translate toward normal direction until MPR's plane is intersect with input point
    void MPRTranslateTo(std::shared_ptr<OrthoCamera> pMPRCamera , const Point3& pt);

    ScanSliceType CheckScaneType(std::shared_ptr<OrthoCamera> pMPRCamera) const;

    Point3 AdjustPointToDiscrete(const Point3& ptWorld) const;

    float ConvertThicknessW2V(std::shared_ptr<OrthoCamera> pMPRCamera , float fThicknessWorldmm) const;

private:
    void Calculate_i();

    void CalculateMatrix_i();

    void CheckVolumeOrthogonal_i();

    void CalculatePatientAxisInfo_i();

    void CalculateVRReplacement_i();

    void CalculateDefaultMPRCenterWorld_i();

    void CaluculateOrthogonalMPRReplacement_i();

private:
    std::shared_ptr<ImageData> m_pImgData;

    bool m_bVolumeOrthogonal;

    PatientAxisInfo m_headInfo;
    PatientAxisInfo m_leftInfo;
    PatientAxisInfo m_posteriorInfo;

    Matrix4 m_matVolume2Physical;
    Matrix4 m_matPhysical2World;
    Matrix4 m_matWorld2Patient;
    Matrix4 m_matVolume2Wolrd;
    Matrix4 m_matWorld2Volume;
    OrthoCamera m_VRPlacementCamera;

    Point3 m_ptDefaultMPRCenter;
    // SAGITTAL= 0, CORONAL = 1, TRANSVERSE = 2,
    OrthoCamera m_OthogonalMPRCamera[3];
    Vector3 m_OthogonalMPRNorm[3];
};

MED_IMAGING_END_NAMESPACE

#endif