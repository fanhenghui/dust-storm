#ifndef MED_IMAGING_RAY_CAST_SCENE_QT_H
#define MED_IMAGING_RAY_CAST_SCENE_QT_H

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class VolumeInfos;
class EntryExitPoints;
class RayCaster;
class RayCasterCanvas;
class CameraCalculator;
class OrthoCameraInteractor;
class OrthoCamera;

class RenderAlgo_Export RayCastScene : public SceneBase
{
public:
    RayCastScene();
    RayCastScene(int iWidth , int iHeight);
    virtual ~RayCastScene();

    virtual void Initialize();
    virtual void Finalize();
    virtual void SetDisplaySize(int iWidth , int iHeight);
    virtual void Render(int iTestCode);

    void SetVolumeInfos(std::shared_ptr<VolumeInfos> pVolumeInfos);

    std::shared_ptr<VolumeInfos> GetVolumeInfos() const;

    std::shared_ptr<CameraCalculator> GetCameraCalculator() const;

    void SetTestCode(int iTestCode);

    //Mask label level
    //Default is L_8
    void SetMaskLabelLevel(LabelLevel eLabelLevel);

    //Sample rate
    void SetSampleRate(float fSampleRate);

    //Label parameter
    void SetVisibleLabels(std::vector<unsigned char> vecLabels);

    //Window level parameter ( unregulated)
    // EG: CT modality , the unit is HU
    void SetWindowlevel(float fWW , float fWL , unsigned char ucLabel);
    void SetGlobalWindowLevel(float fWW , float fWL);
    void GetGlobalWindowLevel(float& fWW , float& fWL) const;

    //Ray casting mode parameter
    void SetMaskMode(MaskMode eMode);
    void SetCompositeMode(CompositeMode eMode);
    void SetInterpolationMode(InterpolationMode eMode);
    void SetShadingMode(ShadingMode eMode);
    void SetColorInverseMode(ColorInverseMode eMode);

protected:
    std::shared_ptr<VolumeInfos> m_pVolumeInfos;
    std::shared_ptr<EntryExitPoints> m_pEntryExitPoints;
    std::shared_ptr<RayCaster> m_pRayCaster;
    std::shared_ptr<RayCasterCanvas> m_pCanvas;

    std::shared_ptr<OrthoCamera> m_pRayCastCamera;

    std::shared_ptr<CameraCalculator> m_pCameraCalculator;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;

    //////////////////////////////////////////////////////////////////////////
    //TODO Temp for test
    float m_fGlobalWW;
    float m_fGlobalWL;
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    //TODO Temp default pseudo color texture 
    //should design a wrap to contain global pseudo colors because its constant
    GLTexture1DPtr m_pPseudoColor;
    int m_iTestCode;

};

MED_IMAGING_END_NAMESPACE
#endif