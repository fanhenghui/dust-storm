#ifndef MEDIMGRENDERALGO_RAY_CAST_SCENE_H
#define MEDIMGRENDERALGO_RAY_CAST_SCENE_H

#include <map>
#include "renderalgo/mi_scene_base.h"

#include "arithmetic/mi_vector2f.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_gpu_resource_pair.h"

MED_IMG_BEGIN_NAMESPACE

class VolumeInfos;
class EntryExitPoints;
class RayCaster;
class RayCasterCanvas;
class CameraCalculator;
class OrthoCameraInteractor;
class OrthoCamera;
class ColorTransFunc;
class OpacityTransFunc;
class GraphicObjectNavigator;
class TransferFunctionTexture;

class RenderAlgo_Export RayCastScene : public SceneBase {
public:
    RayCastScene(RayCastingStrategy strategy, GPUPlatform platfrom);
    RayCastScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom);
    virtual ~RayCastScene();

    virtual void initialize();

    virtual void set_display_size(int width, int height);

    virtual void render();
    virtual void render_to_back();

    virtual void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);
    std::shared_ptr<VolumeInfos> get_volume_infos() const;

    std::shared_ptr<CameraCalculator> get_camera_calculator() const;

    // 0 for ray casting 1 for entry points 2 for exit points
    void set_test_code(int test_code);

    // Mask label level
    // Default is L_8
    void set_mask_label_level(LabelLevel label_level);

    // Sample rate
    void set_sample_step(float sample_step);

    // Label parameter
    virtual void set_visible_labels(std::vector<unsigned char> labels);
    std::vector<unsigned char> get_visible_labels() const;

    //-----------------------------------------------//
    // Window level parameter ( 
    // is unregulated intensity. EG: CT modality, the unit is HU.
    // MIP/MinIP/Average: set WL to global
    // DVR: 
    //    1. mask-none: set WL to label 0
    //    2. mask-multi-label: set WL to label 1 to label_level-1. 
    void set_window_level(float ww, float wl, unsigned char label);
    int  get_window_level(float& ww, float& wl, unsigned char label) const;
    void set_global_window_level(float ww, float wl);
    void get_global_window_level(float& ww, float& wl) const;

    // MinIP threshold(is unregulated intensity)
    void set_minip_threshold(float th);

    // Ray casting mode parameter
    void set_mask_mode(MaskMode mode);
    void set_composite_mode(CompositeMode mode);
    void set_interpolation_mode(InterpolationMode mode);
    void set_shading_mode(ShadingMode mode);
    void set_color_inverse_mode(ColorInverseMode mode);

    MaskMode get_mask_mode() const;
    CompositeMode get_composite_mode() const;
    InterpolationMode get_interpolation_mode() const;
    ShadingMode get_shading_mode() const;
    ColorInverseMode get_color_inverse_mode() const;

    void set_ambient_color(float r, float g, float b, float factor);
    void set_material(const Material& m, unsigned char label);

    // Transfer function
    void set_pseudo_color(std::shared_ptr<ColorTransFunc> color);
    virtual void set_color_opacity(std::shared_ptr<ColorTransFunc> color,
                           std::shared_ptr<OpacityTransFunc> opacity,
                           unsigned char label);

    //return dc coordinate
    Point2 project_point_to_screen(const Point3& pt_w) const;

    virtual void set_downsample(bool flag);

    void set_expected_fps(int fps);
    int get_expected_fps() const;

    //navigator window size : navi_size = min(scene_width, scene_height)/ratio
    //navigator viewport (scene_x - margin_x - navi_size, margin_y, navi_size, navi_size)  
    //default x/y margin(20), default ratio (4.5)
    void set_navigator_visibility(bool flag);
    void set_navigator_para(int x_margin, int y_margin, float ratio);

protected:
    virtual void pre_render();

protected:
    RayCastingStrategy _strategy;
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::shared_ptr<EntryExitPoints> _entry_exit_points;
    std::shared_ptr<RayCaster> _ray_caster;
    std::shared_ptr<RayCasterCanvas> _canvas;

    std::shared_ptr<OrthoCamera> _ray_cast_camera;

    std::shared_ptr<CameraCalculator> _camera_calculator;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

    float _global_ww;
    float _global_wl;
    std::map<unsigned char, Vector2f> _window_levels;

    std::shared_ptr<TransferFunctionTexture> _transfer_funcion_texture;

    //graphic object
    std::shared_ptr<GraphicObjectNavigator> _navigator;
    bool _navigator_vis;
    int  _navigator_margin[2];
    float  _navigator_window_ratio;
};

MED_IMG_END_NAMESPACE
#endif