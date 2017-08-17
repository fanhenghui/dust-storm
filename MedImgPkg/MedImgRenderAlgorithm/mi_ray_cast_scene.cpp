#include "mi_ray_cast_scene.h"

#include "MedImgUtil/mi_configuration.h"
#include "MedImgUtil/mi_file_util.h"

#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_point2.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_texture_1d_array.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_texture_cache.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_camera_calculator.h"
#include "mi_camera_interactor.h"
#include "mi_color_transfer_function.h"
#include "mi_entry_exit_points.h"
#include "mi_opacity_transfer_function.h"
#include "mi_ray_caster.h"
#include "mi_ray_caster_canvas.h"
#include "mi_volume_infos.h"

MED_IMG_BEGIN_NAMESPACE

RayCastScene::RayCastScene() : SceneBase(), _global_ww(0), _global_wl(0) {
  _ray_cast_camera.reset(new OrthoCamera());
  _camera = _ray_cast_camera;

  _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

  _ray_caster.reset(new RayCaster());

  _canvas.reset(new RayCasterCanvas());

  if (CPU == Configuration::instance()->get_processing_unit_type()) {
    _ray_caster->set_strategy(CPU_BASE);
  } else {
    _ray_caster->set_strategy(GPU_BASE);
  }

  init_default_color_texture_i();
}

RayCastScene::RayCastScene(int width, int height)
    : SceneBase(width, height), _global_ww(0), _global_wl(0) {
  _ray_cast_camera.reset(new OrthoCamera());
  _camera = _ray_cast_camera;

  _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

  _ray_caster.reset(new RayCaster());

  _canvas.reset(new RayCasterCanvas());
  _canvas->set_display_size(_width, _height);

  if (CPU == Configuration::instance()->get_processing_unit_type()) {
    _ray_caster->set_strategy(CPU_BASE);
  } else {
    _ray_caster->set_strategy(GPU_BASE);
  }

  init_default_color_texture_i();
}

RayCastScene::~RayCastScene() {}

void RayCastScene::initialize() {
  SceneBase::initialize();

  // Canvas
  _canvas->initialize();
  _entry_exit_points->initialize();
}

void RayCastScene::set_display_size(int width, int height) {
  SceneBase::set_display_size(width, height);
  _canvas->set_display_size(width, height);
  _canvas->update_fbo(); // update texture size
  _entry_exit_points->set_display_size(width, height);
  _camera_interactor->resize(width, height);
}

void RayCastScene::pre_render_i() {
  // refresh volume & mask & their infos
  _volume_infos->refresh();

  // scene FBO , ray casting program ...
  initialize();

  // entry exit points initialize
  _entry_exit_points->initialize();

  // GL resource update (discard)
  GLResourceManagerContainer::instance()->update_all();

  // GL texture udpate
  GLTextureCache::instance()->process_cache();
}

void RayCastScene::init_default_color_texture_i() {
  if (GPU == Configuration::instance()->get_processing_unit_type()) {
    // initialize gray pseudo color texture
    if (!_pseudo_color_texture) {
      UIDType uid;
      _pseudo_color_texture = GLResourceManagerContainer::instance()
                                  ->get_texture_1d_manager()
                                  ->create_object(uid);
      _pseudo_color_texture->set_description("pseudo color texture");
      _res_shield.add_shield<GLTexture1D>(_pseudo_color_texture);

      unsigned char *gray_array = new unsigned char[S_TRANSFER_FUNC_WIDTH * 3];
      for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
        gray_array[i * 3] = (255 - 0) * (float)i / (float)S_TRANSFER_FUNC_WIDTH;
        gray_array[i * 3 + 1] = gray_array[i * 3];
        gray_array[i * 3 + 2] = gray_array[i * 3];
      }
      GLTextureCache::instance()->cache_load(
          GL_TEXTURE_1D, _pseudo_color_texture, GL_CLAMP_TO_EDGE, GL_LINEAR,
          GL_RGB8, S_TRANSFER_FUNC_WIDTH, 0, 0, GL_RGB, GL_UNSIGNED_BYTE,
          (char *)gray_array);
    }

    if (!_color_opacity_texture_array) {
      UIDType uid;
      _color_opacity_texture_array = GLResourceManagerContainer::instance()
                                         ->get_texture_1d_array_manager()
                                         ->create_object(uid);
      _color_opacity_texture_array->set_description(
          "color opacity texture array");

      const int tex_num = 8; // default mask level
      unsigned char *rgba =
          new unsigned char[S_TRANSFER_FUNC_WIDTH * tex_num * 4];
      memset(rgba, 0, S_TRANSFER_FUNC_WIDTH * tex_num * 4);

      GLTextureCache::instance()->cache_load(
          GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array, GL_CLAMP_TO_EDGE,
          GL_LINEAR, GL_RGBA8, S_TRANSFER_FUNC_WIDTH, tex_num, 0, GL_RGBA,
          GL_UNSIGNED_BYTE, (char *)rgba);
    }
  }

  if (CPU == Configuration::instance()->get_processing_unit_type()) {
    // TODO gray pseudo array
  }
}

void RayCastScene::render() {
  std::cout << "in rendering\n";
  pre_render_i();

  // Skip render scene
  if (!get_dirty()) {
    return;
  }

  CHECK_GL_ERROR;

  //////////////////////////////////////////////////////////////////////////
  // TODO other common graphic object rendering list

  //////////////////////////////////////////////////////////////////////////
  // 1 Ray casting
  // glPushAttrib(GL_ALL_ATTRIB_BITS);

  glViewport(0, 0, _width, _height);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);

  _entry_exit_points->calculate_entry_exit_points();

  _ray_caster->render();
  // glPopAttrib();

  //////////////////////////////////////////////////////////////////////////
  // 2 Mapping ray casting result to Scene FBO (<><>flip vertically<><>)
  // glPushAttrib(GL_ALL_ATTRIB_BITS);

  glViewport(0, 0, _width, _height);

  _scene_fbo->bind();
  glDrawBuffer(GL_COLOR_ATTACHMENT0);

  glEnable(GL_TEXTURE_2D);
  _canvas->get_color_attach_texture()->bind();

  glBegin(GL_QUADS);
  // glTexCoord2f(0.0, 0.0);
  // glVertex2f(-1.0, -1.0);
  // glTexCoord2f(1.0, 0.0);
  // glVertex2f(1.0, -1.0);
  // glTexCoord2f(1.0, 1.0);
  // glVertex2f(1.0, 1.0);
  // glTexCoord2f(0.0, 1.0);
  // glVertex2f(-1.0, 1.0);

  glTexCoord2f(0.0, 1.0);
  glVertex2f(-1.0, -1.0);
  glTexCoord2f(1.0, 1.0);
  glVertex2f(1.0, -1.0);
  glTexCoord2f(1.0, 0.0);
  glVertex2f(1.0, 1.0);
  glTexCoord2f(0.0, 0.0);
  glVertex2f(-1.0, 1.0);

  glEnd();

  // CHECK_GL_ERROR;
  // glPopAttrib();//TODO Here will give a GL_INVALID_OPERATION error !!!
  // CHECK_GL_ERROR;

  _scene_fbo->unbind();

  CHECK_GL_ERROR;

  // CHECK_GL_ERROR;
  //_canvas->debug_output_color("/home/wr/data/output.raw");
  // CHECK_GL_ERROR;

  // TODO Test code
  {
    FBOStack fbo_stack;
    _scene_color_attach_0->bind();
    unsigned char *color_array = new unsigned char[_width * _height * 3];
    _scene_color_attach_0->download(GL_RGB, GL_UNSIGNED_BYTE, color_array);
    std::stringstream ss;
    ss << "/home/wr/data/scene_img_" << _name << "_" << _width << "_" << _height
       << ".rgb";
    FileUtil::write_raw(ss.str(), (char *)color_array, _width * _height * 3);
  }

  set_dirty(false);

  std::cout << "out rendering\n";
}

void RayCastScene::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos) {
  try {
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_infos);
    _volume_infos = volume_infos;

    std::shared_ptr<ImageData> volume = _volume_infos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume);

    std::shared_ptr<ImageData> mask = _volume_infos->get_mask();
    RENDERALGO_CHECK_NULL_EXCEPTION(mask);

    std::shared_ptr<ImageDataHeader> data_header =
        _volume_infos->get_data_header();
    RENDERALGO_CHECK_NULL_EXCEPTION(data_header);

    // Camera calculator
    _camera_calculator = volume_infos->get_camera_calculator();

    // Entry exit points
    _entry_exit_points->set_volume_data(volume);
    _entry_exit_points->set_camera(_camera);
    _entry_exit_points->set_display_size(_width, _height);
    _entry_exit_points->set_camera_calculator(_camera_calculator);

    // Ray caster
    _ray_caster->set_canvas(_canvas);
    _ray_caster->set_entry_exit_points(_entry_exit_points);
    _ray_caster->set_camera(_camera);
    _ray_caster->set_camera_calculator(_camera_calculator);
    _ray_caster->set_volume_data(volume);
    _ray_caster->set_mask_data(mask);

    if (GPU == Configuration::instance()->get_processing_unit_type()) {
      // set texture
      _ray_caster->set_pseudo_color_texture(_pseudo_color_texture,
                                            S_TRANSFER_FUNC_WIDTH);
      _ray_caster->set_color_opacity_texture_array(
          _color_opacity_texture_array);
      _ray_caster->set_volume_data_texture(volume_infos->get_volume_texture());
      _ray_caster->set_mask_data_texture(volume_infos->get_mask_texture());
    }

    set_dirty(true);
  } catch (const Exception &e) {
    // TOOD LOG
    std::cout << e.what();
    // assert(false);
    throw e;
  }
}

void RayCastScene::set_mask_label_level(LabelLevel label_level) {
  _ray_caster->set_mask_label_level(label_level);

  if (GPU == Configuration::instance()->get_processing_unit_type()) {
    if (!_color_opacity_texture_array) {
      UIDType uid;
      _color_opacity_texture_array = GLResourceManagerContainer::instance()
                                         ->get_texture_1d_array_manager()
                                         ->create_object(uid);
      _color_opacity_texture_array->set_description(
          "color opacity texture array");
    }

    const int tex_num = static_cast<int>(label_level);
    unsigned char *rgba =
        new unsigned char[S_TRANSFER_FUNC_WIDTH * tex_num * 4];
    memset(rgba, 0, S_TRANSFER_FUNC_WIDTH * tex_num * 4);

    // reshape color opacity texture array
    GLTextureCache::instance()->cache_load(
        GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array, GL_CLAMP_TO_EDGE,
        GL_LINEAR, GL_RGBA8, S_TRANSFER_FUNC_WIDTH, tex_num, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, (char *)rgba);
  }

  set_dirty(true);
}

void RayCastScene::set_sample_rate(float sample_rate) {
  _ray_caster->set_sample_rate(sample_rate);
  set_dirty(true);
}

void RayCastScene::set_visible_labels(std::vector<unsigned char> labels) {
  for (auto it = labels.begin(); it != labels.end(); ++it) {
    if (*it == 0) {
      RENDERALGO_THROW_EXCEPTION("visible labels contain zero");
    }
  }
  if (_ray_caster->get_visible_labels() != labels) {
    _ray_caster->set_visible_labels(labels);
    set_dirty(true);
  }
}

void RayCastScene::set_window_level(float ww, float wl, unsigned char label) {
  RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
  _volume_infos->get_volume()->regulate_normalize_wl(ww, wl);

  _ray_caster->set_window_level(ww, wl, label);

  set_dirty(true);
}

void RayCastScene::set_global_window_level(float ww, float wl) {
  _global_ww = ww;
  _global_wl = wl;

  RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
  _volume_infos->get_volume()->regulate_wl(ww, wl);

  _ray_caster->set_global_window_level(ww, wl);

  set_dirty(true);
}

void RayCastScene::set_mask_mode(MaskMode mode) {
  if (_ray_caster->get_mask_mode() != mode) {
    _ray_caster->set_mask_mode(mode);
    set_dirty(true);
  }
}

void RayCastScene::set_composite_mode(CompositeMode mode) {
  if (_ray_caster->get_composite_mode() != mode) {
    _ray_caster->set_composite_mode(mode);
    set_dirty(true);
  }
}

void RayCastScene::set_interpolation_mode(InterpolationMode mode) {
  if (_ray_caster->get_interpolation_mode() != mode) {
    _ray_caster->set_interpolation_mode(mode);
    set_dirty(true);
  }
}

void RayCastScene::set_shading_mode(ShadingMode mode) {
  if (_ray_caster->get_shading_mode() != mode) {
    _ray_caster->set_shading_mode(mode);
    set_dirty(true);
  }
}

void RayCastScene::set_color_inverse_mode(ColorInverseMode mode) {
  if (_ray_caster->get_color_inverse_mode() != mode) {
    _ray_caster->set_color_inverse_mode(mode);
    set_dirty(true);
  }
}

void RayCastScene::set_ambient_color(float r, float g, float b, float factor) {
  _ray_caster->set_ambient_color(r, g, b, factor);
  set_dirty(true);
}

void RayCastScene::set_material(const Material &m, unsigned char label) {
  _ray_caster->set_material(m, label);
  set_dirty(true);
}

void RayCastScene::set_pseudo_color(std::shared_ptr<ColorTransFunc> color) {
  if (GPU == Configuration::instance()->get_processing_unit_type()) {
    RENDERALGO_CHECK_NULL_EXCEPTION(_pseudo_color_texture);

    std::vector<ColorTFPoint> pts;
    color->set_width(S_TRANSFER_FUNC_WIDTH);
    color->get_point_list(pts);
    unsigned char *rgb = new unsigned char[S_TRANSFER_FUNC_WIDTH * 3];
    for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
      rgb[i * 3] = static_cast<unsigned char>(pts[i].x);
      rgb[i * 3 + 1] = static_cast<unsigned char>(pts[i].y);
      rgb[i * 3 + 2] = static_cast<unsigned char>(pts[i].z);
    }

    GLTextureCache::instance()->cache_update(
        GL_TEXTURE_1D, _pseudo_color_texture, 0, 0, 0, S_TRANSFER_FUNC_WIDTH, 0,
        0, GL_RGB, GL_UNSIGNED_BYTE, (char *)rgb);
  }

  set_dirty(true);
}

void RayCastScene::set_color_opacity(std::shared_ptr<ColorTransFunc> color,
                                     std::shared_ptr<OpacityTransFunc> opacity,
                                     unsigned char label) {
  if (GPU == Configuration::instance()->get_processing_unit_type()) {
    std::vector<ColorTFPoint> color_pts;
    color->set_width(S_TRANSFER_FUNC_WIDTH);
    color->get_point_list(color_pts);

    std::vector<OpacityTFPoint> opacity_pts;
    opacity->set_width(S_TRANSFER_FUNC_WIDTH);
    opacity->get_point_list(opacity_pts);

    unsigned char *rgba = new unsigned char[S_TRANSFER_FUNC_WIDTH * 4];
    for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
      rgba[i * 4] = static_cast<unsigned char>(color_pts[i].x);
      rgba[i * 4 + 1] = static_cast<unsigned char>(color_pts[i].y);
      rgba[i * 4 + 2] = static_cast<unsigned char>(color_pts[i].z);
      rgba[i * 4 + 3] = static_cast<unsigned char>(opacity_pts[i].a);
    }

    GLTextureCache::instance()->cache_update(
        GL_TEXTURE_1D_ARRAY, _color_opacity_texture_array, 0, label, 0,
        S_TRANSFER_FUNC_WIDTH, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, (char *)rgba);
  }

  set_dirty(true);
}

void RayCastScene::set_test_code(int test_code) {
  _ray_caster->set_test_code(test_code);

  set_dirty(true);
}

void RayCastScene::get_global_window_level(float &ww, float &wl) const {
  ww = _global_ww;
  wl = _global_wl;
}

std::shared_ptr<VolumeInfos> RayCastScene::get_volume_infos() const {
  return _volume_infos;
}

std::shared_ptr<CameraCalculator> RayCastScene::get_camera_calculator() const {
  return _camera_calculator;
}

MED_IMG_END_NAMESPACE