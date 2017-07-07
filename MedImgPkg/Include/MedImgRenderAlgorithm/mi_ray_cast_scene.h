#ifndef MED_IMG_RAY_CAST_SCENE_QT_H
#define MED_IMG_RAY_CAST_SCENE_QT_H

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE

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
    RayCastScene(int width , int height);
    virtual ~RayCastScene();

    virtual void initialize();
    virtual void finalize();

    virtual void set_display_size(int width , int height);

    virtual void render(int test_code);

    void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);
    std::shared_ptr<VolumeInfos> get_volume_infos() const;

    std::shared_ptr<CameraCalculator> get_camera_calculator() const;

    void set_test_code(int test_code);

    //Mask label level
    //Default is L_8
    void set_mask_label_level(LabelLevel label_level);

    //Sample rate
    void set_sample_rate(float sample_rate);

    //Label parameter
    void set_visible_labels(std::vector<unsigned char> labels);

    //Window level parameter ( unregulated)
    // EG: CT modality , the unit is HU
    void set_window_level(float ww , float wl , unsigned char label);
    void set_global_window_level(float ww , float wl);
    void get_global_window_level(float& ww , float& wl) const;

    //Ray casting mode parameter
    void set_mask_mode(MaskMode mode);
    void set_composite_mode(CompositeMode mode);
    void set_interpolation_mode(InterpolationMode mode);
    void set_shading_mode(ShadingMode mode);
    void set_color_inverse_mode(ColorInverseMode mode);


    void debug_output_entry_points(const std::string& file_name);
    void debug_output_exit_points(const std::string& file_name);

protected:
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::shared_ptr<EntryExitPoints> _entry_exit_points;
    std::shared_ptr<RayCaster> _ray_caster;
    std::shared_ptr<RayCasterCanvas> _canvas;

    std::shared_ptr<OrthoCamera> _ray_cast_camera;

    std::shared_ptr<CameraCalculator> _camera_calculator;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

    //////////////////////////////////////////////////////////////////////////
    //TODO Temp for test
    float _global_ww;
    float _global_wl;
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    //TODO Temp default pseudo color texture 
    //should design a wrap to contain global pseudo colors because its constant
    GLTexture1DPtr _pseudo_color_texture;
    int _test_code;

};

MED_IMG_END_NAMESPACE
#endif