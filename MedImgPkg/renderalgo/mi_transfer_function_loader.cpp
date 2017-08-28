#include "mi_transfer_function_loader.h"

#include "util//mi_string_number_converter.h"

#include "mi_color_transfer_function.h"
#include "mi_opacity_transfer_function.h"

#ifdef WIN32
#include "pugixml/pugiconfig.hpp"
#include "pugixml/pugixml.hpp"
#else
#include "pugiconfig.hpp"
#include "pugixml.hpp"
#endif

MED_IMG_BEGIN_NAMESPACE

IOStatus
TransferFuncLoader::load_pseudo_color(const std::string &xml,
                                      std::shared_ptr<ColorTransFunc> &color) {
  pugi::xml_document doc;
  if (!doc.load_file(xml.c_str())) {
    return IO_FILE_OPEN_FAILED;
  }

  pugi::xml_node root = doc.child("Lut");
  if (root.empty()) {
    return IO_DATA_DAMAGE;
  }

  color.reset(new ColorTransFunc());
  color->set_color_type(RGB, RGB);

  std::string name = root.attribute("name").as_string();
  color->set_name(name);

  pugi::xpath_node_set points = root.select_nodes("point");
  const size_t pt_num = points.size();
  if (pt_num < 2) {
    return IO_DATA_DAMAGE;
  }

  StrNumConverter<float> conv;
  for (auto it = points.begin(); it != points.end(); ++it) {
    pugi::xml_node node = (*it).node();
    pugi::xml_node idx_node = node.child("index");
    pugi::xml_node r_node = node.child("R");
    pugi::xml_node g_node = node.child("G");
    pugi::xml_node b_node = node.child("B");
    if (idx_node.empty() || r_node.empty() || g_node.empty() ||
        b_node.empty()) {
      return IO_DATA_DAMAGE;
    }

    float idx = conv.to_num(idx_node.child_value());
    float r = conv.to_num(r_node.child_value());
    float g = conv.to_num(g_node.child_value());
    float b = conv.to_num(b_node.child_value());

    color->add_rgb_point(idx, r, g, b);
  }

  return IO_SUCCESS;
}

IOStatus TransferFuncLoader::load_color_opacity(
    const std::string &xml, std::shared_ptr<ColorTransFunc> &color,
    std::shared_ptr<OpacityTransFunc> &opacity, float &ww, float &wl,
    RGBAUnit &background, Material &material) {
  pugi::xml_document doc;
  if (!doc.load_file(xml.c_str())) {
    return IO_FILE_OPEN_FAILED;
  }

  pugi::xml_node root = doc.child("Lut");
  if (root.empty()) {
    return IO_DATA_DAMAGE;
  }

  color.reset(new ColorTransFunc());
  opacity.reset(new OpacityTransFunc());

  const std::string name = root.attribute("name").as_string();
  color->set_name(name);
  opacity->set_name(name);

  // color
  pugi::xml_node color_node = root.child("Color");
  if (color_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  const std::string color_inter =
      color_node.attribute("Interpolation").as_string();
  if (color_inter == "RGB") {
    color->set_color_type(RGB, RGB);
  } else if (color_inter == "HSV") {
    color->set_color_type(RGB, HSV);
  }

  pugi::xpath_node_set points = color_node.select_nodes("point");
  size_t pt_num = points.size();
  if (pt_num < 2) {
    return IO_DATA_DAMAGE;
  }

  StrNumConverter<float> conv;
  for (auto it = points.begin(); it != points.end(); ++it) {
    pugi::xml_node node = (*it).node();
    pugi::xml_node idx_node = node.child("index");
    pugi::xml_node r_node = node.child("R");
    pugi::xml_node g_node = node.child("G");
    pugi::xml_node b_node = node.child("B");
    if (idx_node.empty() || r_node.empty() || g_node.empty() ||
        b_node.empty()) {
      return IO_DATA_DAMAGE;
    }

    float idx = conv.to_num(idx_node.child_value());
    float r = conv.to_num(r_node.child_value());
    float g = conv.to_num(g_node.child_value());
    float b = conv.to_num(b_node.child_value());

    color->add_rgb_point(idx, r, g, b);
  }

  // Opacity
  pugi::xml_node opacity_node = root.child("Opacity");
  if (opacity_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  points = opacity_node.select_nodes("point");
  pt_num = points.size();
  if (pt_num < 2) {
    return IO_DATA_DAMAGE;
  }

  for (auto it = points.begin(); it != points.end(); ++it) {
    pugi::xml_node node = (*it).node();
    pugi::xml_node idx_node = node.child("index");
    pugi::xml_node a_node = node.child("A");
    if (idx_node.empty() || a_node.empty()) {
      return IO_DATA_DAMAGE;
    }

    float idx = conv.to_num(idx_node.child_value());
    float a = conv.to_num(a_node.child_value());

    opacity->add_point(idx, a);
  }

  // WL
  pugi::xml_node wl_node = root.child("WindowLevel");
  if (wl_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  pugi::xml_node width_node = wl_node.child("width");
  pugi::xml_node level_node = wl_node.child("level");
  if (width_node.empty() || level_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  ww = conv.to_num(width_node.child_value());
  wl = conv.to_num(level_node.child_value());

  // Background
  pugi::xml_node bg_node = root.child("Background");
  if (bg_node.empty()) {
    return IO_DATA_DAMAGE;
  }
  pugi::xml_node bg_r = bg_node.child("R");
  pugi::xml_node bg_g = bg_node.child("G");
  pugi::xml_node bg_b = bg_node.child("B");
  pugi::xml_node bg_a = bg_node.child("A");
  if (bg_r.empty() || bg_g.empty() || bg_b.empty() || bg_a.empty()) {
    return IO_DATA_DAMAGE;
  }

  background.r = static_cast<unsigned char>(conv.to_num(bg_r.child_value()));
  background.g = static_cast<unsigned char>(conv.to_num(bg_g.child_value()));
  background.b = static_cast<unsigned char>(conv.to_num(bg_b.child_value()));
  background.a = static_cast<unsigned char>(conv.to_num(bg_a.child_value()));

  // Material
  pugi::xml_node material_node = root.child("Material");
  if (material_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  pugi::xml_node diffuse_node = material_node.child("DiffuseColor");
  if (diffuse_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  pugi::xml_node specular_node = material_node.child("SpecularColor");
  if (specular_node.empty()) {
    return IO_DATA_DAMAGE;
  }

  pugi::xml_node diffuse_r = diffuse_node.child("R");
  pugi::xml_node diffuse_g = diffuse_node.child("G");
  pugi::xml_node diffuse_b = diffuse_node.child("B");
  pugi::xml_node diffuse_factor = diffuse_node.child("Factor");
  if (diffuse_r.empty() || diffuse_g.empty() || diffuse_b.empty() ||
      diffuse_factor.empty()) {
    return IO_DATA_DAMAGE;
  }

  material.diffuse[0] = conv.to_num(diffuse_r.child_value()) / 255.0f;
  material.diffuse[1] = conv.to_num(diffuse_g.child_value()) / 255.0f;
  material.diffuse[2] = conv.to_num(diffuse_b.child_value()) / 255.0f;
  material.diffuse[3] = conv.to_num(diffuse_factor.child_value());

  pugi::xml_node specular_r = specular_node.child("R");
  pugi::xml_node specular_g = specular_node.child("G");
  pugi::xml_node specular_b = specular_node.child("B");
  pugi::xml_node specular_factor = specular_node.child("Factor");
  pugi::xml_node specular_shininess = specular_node.child("Shininess");
  if (specular_r.empty() || specular_g.empty() || specular_b.empty() ||
      specular_factor.empty() || specular_shininess.empty()) {
    return IO_DATA_DAMAGE;
  }

  material.specular[0] = conv.to_num(specular_r.child_value()) / 255.0f;
  material.specular[1] = conv.to_num(specular_g.child_value()) / 255.0f;
  material.specular[2] = conv.to_num(specular_b.child_value()) / 255.0f;
  material.specular[3] = conv.to_num(specular_factor.child_value());
  material.specular_shiness = conv.to_num(specular_shininess.child_value());

  return IO_SUCCESS;
}

MED_IMG_END_NAMESPACE