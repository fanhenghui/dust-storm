#include "mi_entry_exit_points.h"

#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgArithmetic/mi_camera_base.h"

#include "MedImgUtil/mi_file_util.h"

#include "mi_camera_calculator.h"

MED_IMG_BEGIN_NAMESPACE

EntryExitPoints::EntryExitPoints()
    : _width(4), _height(4), _has_init(false), _strategy(CPU_BASE) {
  _entry_points_buffer.reset(new Vector4f[_width * _height]);
  _exit_points_buffer.reset(new Vector4f[_width * _height]);

  UIDType uid;
  _entry_points_texture = GLResourceManagerContainer::instance()
                              ->get_texture_2d_manager()
                              ->create_object(uid);
  _entry_points_texture->set_description("entry points texture");
  _exit_points_texture = GLResourceManagerContainer::instance()
                             ->get_texture_2d_manager()
                             ->create_object(uid);
  _exit_points_texture->set_description("exit points texture");

  _res_shield.add_shield<GLTexture2D>(_entry_points_texture);
  _res_shield.add_shield<GLTexture2D>(_exit_points_texture);
}

void EntryExitPoints::initialize() {
  if (!_has_init) {
    _entry_points_texture->initialize();
    _exit_points_texture->initialize();

    _entry_points_texture->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _entry_points_texture->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT,
                                NULL);
    _entry_points_texture->unbind();

    _exit_points_texture->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _exit_points_texture->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT,
                               NULL);
    _exit_points_texture->unbind();

    _has_init = true;
  }
}

EntryExitPoints::~EntryExitPoints() {}

void EntryExitPoints::set_display_size(int width, int height) {
  _width = width;
  _height = height;
  _entry_points_buffer.reset(new Vector4f[_width * _height]);
  _exit_points_buffer.reset(new Vector4f[_width * _height]);

  // resize texture
  if (GPU_BASE == _strategy && _has_init) {
    CHECK_GL_ERROR;

    _entry_points_texture->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _entry_points_texture->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT,
                                NULL);
    _entry_points_texture->unbind();

    _exit_points_texture->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _exit_points_texture->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT,
                               NULL);
    _exit_points_texture->unbind();

    CHECK_GL_ERROR;
  }
}

void EntryExitPoints::get_display_size(int &width, int &height) {
  width = _width;
  height = _height;
}

GLTexture2DPtr EntryExitPoints::get_entry_points_texture() {
  return _entry_points_texture;
}

GLTexture2DPtr EntryExitPoints::get_exit_points_texture() {
  return _exit_points_texture;
}

Vector4f *EntryExitPoints::get_entry_points_array() {
  return _entry_points_buffer.get();
}

Vector4f *EntryExitPoints::get_exit_points_array() {
  return _exit_points_buffer.get();
}

void EntryExitPoints::set_volume_data(std::shared_ptr<ImageData> image_data) {
  _volume_data = image_data;
}

std::shared_ptr<ImageData> EntryExitPoints::get_volume_data() const {
  return _volume_data;
}

void EntryExitPoints::set_camera(std::shared_ptr<CameraBase> camera) {
  _camera = camera;
}

std::shared_ptr<CameraBase> EntryExitPoints::get_camera() const {
  return _camera;
}

void EntryExitPoints::set_camera_calculator(
    std::shared_ptr<CameraCalculator> camera_cal) {
  _camera_calculator = camera_cal;
}

std::shared_ptr<CameraCalculator>
EntryExitPoints::get_camera_calculator() const {
  return _camera_calculator;
}

void EntryExitPoints::debug_output_entry_points(const std::string &file_name) {
  Vector4f *pPoints = _entry_points_buffer.get();

  std::unique_ptr<unsigned char[]> rgb_array(
      new unsigned char[_width * _height * 3]);
  RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);
  unsigned int *dim = _volume_data->_dim;
  float dim_r[3] = {1.0f / (float)dim[0], 1.0f / (float)dim[1],
                    1.0f / (float)dim[2]};
  unsigned char r, g, b;
  float rr, gg, bb;
  for (int i = 0; i < _width * _height; ++i) {
    rr = pPoints[i]._m[0] * dim_r[0] * 255.0f;
    gg = pPoints[i]._m[1] * dim_r[1] * 255.0f;
    bb = pPoints[i]._m[2] * dim_r[2] * 255.0f;

    rr = rr > 255.0f ? 255.0f : rr;
    rr = rr < 0.0f ? 0.0f : rr;

    gg = gg > 255.0f ? 255.0f : gg;
    gg = gg < 0.0f ? 0.0f : gg;

    bb = bb > 255.0f ? 255.0f : bb;
    bb = bb < 0.0f ? 0.0f : bb;

    r = static_cast<unsigned char>(rr);
    g = static_cast<unsigned char>(gg);
    b = static_cast<unsigned char>(bb);

    rgb_array[i * 3] = r;
    rgb_array[i * 3 + 1] = g;
    rgb_array[i * 3 + 2] = b;
  }

  FileUtil::write_raw(file_name, rgb_array.get(), _width * _height * 3);
}

void EntryExitPoints::debug_output_exit_points(const std::string &file_name) {
  Vector4f *pPoints = _exit_points_buffer.get();
  std::unique_ptr<unsigned char[]> rgb_array(
      new unsigned char[_width * _height * 3]);

  RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);
  unsigned int *dim = _volume_data->_dim;
  float dim_r[3] = {1.0f / (float)dim[0], 1.0f / (float)dim[1],
                    1.0f / (float)dim[2]};
  unsigned char r, g, b;
  float rr, gg, bb;
  for (int i = 0; i < _width * _height; ++i) {
    rr = pPoints[i]._m[0] * dim_r[0] * 255.0f;
    gg = pPoints[i]._m[1] * dim_r[1] * 255.0f;
    bb = pPoints[i]._m[2] * dim_r[2] * 255.0f;

    rr = rr > 255.0f ? 255.0f : rr;
    rr = rr < 0.0f ? 0.0f : rr;

    gg = gg > 255.0f ? 255.0f : gg;
    gg = gg < 0.0f ? 0.0f : gg;

    bb = bb > 255.0f ? 255.0f : bb;
    bb = bb < 0.0f ? 0.0f : bb;

    r = static_cast<unsigned char>(rr);
    g = static_cast<unsigned char>(gg);
    b = static_cast<unsigned char>(bb);

    rgb_array[i * 3] = r;
    rgb_array[i * 3 + 1] = g;
    rgb_array[i * 3 + 2] = b;
  }

  FileUtil::write_raw(file_name, rgb_array.get(), _width * _height * 3);
}

void EntryExitPoints::set_strategy(RayCastingStrategy strategy) {
  _strategy = strategy;
}

MED_IMG_END_NAMESPACE