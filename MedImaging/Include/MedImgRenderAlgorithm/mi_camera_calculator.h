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

//volume����ϵ������
enum CoordinateAxis
{
    POSX = 0,
    NEGX = 1,
    POSY = 2,
    NEGY = 3,
    POSZ = 4,
    NEGZ = 5,
};

//���˷�λ��Ϣ
struct PatientAxisInfo
{
    //���patient����ϵ ��� �� volume����ϵ�᷽��
    //EG��PatientAxisInfo leftAxisInfo; leftAxisInfo.eVolumeCoord = POSX ˵����������ϵLeft���������������ϵ����X����
    CoordinateAxis eVolumeCoord;

    //��patient����ϵ ��� �� volume����ϵ�᷽�� ��ʵ��patient����
    //EG : ͬ�ϣ�leftAxisInfo.vecPatient = (0.9 0 0) , ˵���ͱ�׼�� (1,0,0) ��ƫ��
    // ����Ϊ����world ����ϵ�£�����Ͳ�������ϵ��ͬ���ܹ����������λ�ã�δ�غͲ�������ϵ��ƽ�У����۲�volume��������Ϊ�˿�ɨ��λ��������patient�����������
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

    //�����ı��˶�������pixel �� mm �������������ϵ�µ�Volume bounding box��Ȼ��axis aligned
    const Matrix4& GetVolumeToPhysicalMatrix() const;

    //��ת���˺Ͳ�������ϵ����룬��������ϵ��ԭ�����������������ĵ�
    const Matrix4& GetPhysicalToWorldMatrix() const;

    const Matrix4& GetVolumeToWorldMatrix() const;

    const Matrix4& GetWorldToVolumeMatrix() const;

    //��������ϵ����������ϵ����������ԭ�㲻һ������
    const Matrix4& GetWorldToPatientMatrix() const;

    const Point3& GetDefaultMPRCenterWorld() const;

    void InitMPRPlacement(std::shared_ptr<OrthoCamera> pCamera , ScanSliceType eScanSliceType , const Point3& ptWorldCenterPoint) const;//ʵʱ��ȡ

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