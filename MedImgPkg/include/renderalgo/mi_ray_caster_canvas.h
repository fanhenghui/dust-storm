#ifndef MEDIMGRENDERALGO_RAY_CASTER_CANVAS_H
#define MEDIMGRENDERALGO_RAY_CASTER_CANVAS_H

#include "arithmetic/mi_color_unit.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "io/mi_io_define.h"
#include "renderalgo/mi_render_algo_export.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export RayCasterCanvas {
public:
  RayCasterCanvas();

  ~RayCasterCanvas();

  void initialize();

  void set_display_size(int width, int height);

  void get_display_size(int &width, int &height) const;

  GLFBOPtr get_fbo();

  GLTexture2DPtr get_color_attach_texture();

  RGBAUnit *get_color_array();

  void update_color_array();

public:
  void debug_output_color(const std::string &file_name);

protected:
private:
  GLFBOPtr _gl_fbo;
  GLTexture2DPtr _color_attach_0; // For RGBA Color
  GLTexture2DPtr _depth_attach;
  GLResourceShield _res_shield;

  int _width;
  int _height;
  std::unique_ptr<RGBAUnit[]> _color_array;
  bool _has_init;
};

MED_IMG_END_NAMESPACE

#endif