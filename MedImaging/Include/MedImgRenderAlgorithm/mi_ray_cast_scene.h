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

    virtual void initialize();
    virtual void finalize();
    virtual void set_display_size(int iWidth , int iHeight);
    virtual void render(int iTestCode);

    void set_volume_infos(std::shared_ptr<VolumeInfos> pVolumeInfos);

    std::shared_ptr<VolumeInfos> get_volume_infos() const;

    std::shared_ptr<CameraCalculator> get_camera_calculator() const;

    void set_test_code(int iTestCode);

    //Mask label level
    //Default is L_8
    void set_mask_label_level(LabelLevel eLabelLevel);

    //Sample rate
    void set_sample_rate(float fSampleRate);

    //Label parameter
    void set_visible_labels(std::vector<unsigned char> vecLabels);

    //Window level parameter ( unregulated)
    // EG: CT modality , the unit is HU
    void set_window_level(float fWW , float fWL , unsigned char ucLabel);
    void set_global_window_level(float fWW , float fWL);
    void get_global_window_level(float& fWW , float& fWL) const;

    //Ray casting mode parameter
    void set_mask_mode(MaskMode eMode);
    void set_composite_mode(CompositeMode eMode);
    void set_interpolation_mode(InterpolationMode eMode);
    void set_shading_mode(ShadingMode eMode);
    void set_color_inverse_mode(ColorInverseMode eMode);

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