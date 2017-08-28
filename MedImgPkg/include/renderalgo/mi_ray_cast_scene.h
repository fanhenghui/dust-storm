#ifndef MEDIMGRENDERALGO_RAY_CAST_SCENE_H
#define MEDIMGRENDERALGO_RAY_CAST_SCENE_H

#include "arithmetic/mi_vector2f.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_scene_base.h"
#include <map>

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

class RenderAlgo_Export RayCastScene : public SceneBase {
public:
  RayCastScene();
  RayCastScene(int width, int height);
  virtual ~RayCastScene();

  virtual void initialize();

  virtual void set_display_size(int width, int height);

  virtual void render();

  virtual void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);
  std::shared_ptr<VolumeInfos> get_volume_infos() const;

  std::shared_ptr<CameraCalculator> get_camera_calculator() const;

  // 0 for ray casting 1 for entry points 2 for exit points
  void set_test_code(int test_code);

  // Mask label level
  // Default is L_8
  void set_mask_label_level(LabelLevel label_level);

  // Sample rate
  void set_sample_rate(float sample_rate);

  // Label parameter
  void set_visible_labels(std::vector<unsigned char> labels);

  // Window level parameter ( unregulated)
  // EG: CT modality , the unit is HU
  virtual void set_window_level(float ww, float wl, unsigned char label);
  int get_window_level(float &ww, float &wl, unsigned char label) const;
  virtual void set_global_window_level(float ww, float wl);
  void get_global_window_level(float &ww, float &wl) const;

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
  void set_material(const Material &m, unsigned char label);

  // Transfer function
  void set_pseudo_color(std::shared_ptr<ColorTransFunc> color);
  void set_color_opacity(std::shared_ptr<ColorTransFunc> color,
                         std::shared_ptr<OpacityTransFunc> opacity,
                         unsigned char label);

protected:
  virtual void pre_render_i();
  void init_default_color_texture_i();

protected:
  std::shared_ptr<VolumeInfos> _volume_infos;
  std::shared_ptr<EntryExitPoints> _entry_exit_points;
  std::shared_ptr<RayCaster> _ray_caster;
  std::shared_ptr<RayCasterCanvas> _canvas;

  std::shared_ptr<OrthoCamera> _ray_cast_camera;

  std::shared_ptr<CameraCalculator> _camera_calculator;
  std::shared_ptr<OrthoCameraInteractor> _camera_interactor;

  //////////////////////////////////////////////////////////////////////////
  // TODO Temp for test
  float _global_ww;
  float _global_wl;
  std::map<unsigned char, Vector2f> _window_levels;
  //////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////
  // should design a wrap to contain global pseudo colors because its
  // constant
  GLTexture1DPtr _pseudo_color_texture;
  GLTexture1DArrayPtr _color_opacity_texture_array;
};

MED_IMG_END_NAMESPACE
#endif