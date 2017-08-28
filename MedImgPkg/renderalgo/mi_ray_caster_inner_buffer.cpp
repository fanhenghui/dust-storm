#include "mi_ray_caster_inner_buffer.h"
#include "glresource/mi_gl_buffer.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_utils.h"

MED_IMG_BEGIN_NAMESPACE
struct RayCasterInnerBuffer::GLResource {
  std::map<RayCasterInnerBuffer::BufferType, GLBufferPtr> buffer_ids;
  bool dirty_flag[TYPE_END];
  GLResourceShield res_shield;

  GLResource() { memset(dirty_flag, 1, sizeof(bool) * TYPE_END); }

  void release() {
    buffer_ids.clear();
    memset(dirty_flag, 0, sizeof(bool) * TYPE_END);
  }

  std::string get_buffer_type_name(RayCasterInnerBuffer::BufferType type) {
    switch (type) {
    case WINDOW_LEVEL_BUCKET: {
      return "window level bucket";
    }
    case VISIBLE_LABEL_BUCKET: {
      return "visible label bucket";
    }
    case VISIBLE_LABEL_ARRAY: {
      return "visible label array";
    }
    case MASK_OVERLAY_COLOR_BUCKET: {
      return "mask overlay color bucket";
    }
    case MATERIAL_BUCKET: {
      return "material bucket";
    }
    default: { return "undefined buffer type"; }
    }
  }

  GLBufferPtr GetBuffer(RayCasterInnerBuffer::BufferType type) {
    auto it = buffer_ids.find(type);
    if (it != buffer_ids.end()) {
      return it->second;
    } else {
      UIDType buffer_id = 0;
      GLBufferPtr buffer = GLResourceManagerContainer::instance()
                               ->get_buffer_manager()
                               ->create_object(buffer_id);
      buffer->set_description("ray caster inner buffer : " +
                              get_buffer_type_name(type));
      buffer->initialize();
      buffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
      buffer_ids[type] = buffer;
      res_shield.add_shield<GLBuffer>(buffer);
      return buffer;
    }
  }
};

RayCasterInnerBuffer::RayCasterInnerBuffer()
    : _inner_resource(new GLResource()), _label_level(L_8) {
  memset(_inner_resource->dirty_flag, 1, sizeof(bool) * TYPE_END);
  _shared_buffer_array.reset(
      new char[static_cast<int>(_label_level) * sizeof(Material)]);
}

RayCasterInnerBuffer::~RayCasterInnerBuffer() {}

GLBufferPtr RayCasterInnerBuffer::get_buffer(BufferType type) {
  try {
    GLBufferPtr buffer = _inner_resource->GetBuffer(type);

    CHECK_GL_ERROR;

    switch (type) {
    case WINDOW_LEVEL_BUCKET: {
      if (_inner_resource->dirty_flag[type]) {
        float *wl_array = (float *)_shared_buffer_array.get();
        memset(wl_array, 0, sizeof(float) * static_cast<int>(_label_level) * 2);

        for (auto it = _window_levels.begin(); it != _window_levels.end();
             ++it) {
          const unsigned char label = it->first;
          if (label > static_cast<int>(_label_level) - 1) {
            std::stringstream ss;
            ss << "Input window level label : " << (int)(label)
               << " is greater than the limit : "
               << static_cast<int>(_label_level) - 1 << " !";
            RENDERALGO_THROW_EXCEPTION(ss.str());
          }
          wl_array[label * 2] = it->second._value.x;
          wl_array[label * 2 + 1] = it->second._value.y;
        }

        buffer->bind();
        buffer->load(static_cast<int>(_label_level) * sizeof(float) * 2,
                     wl_array, GL_STATIC_DRAW);

        _inner_resource->dirty_flag[type] = false;
      }
      break;
    }

    case VISIBLE_LABEL_BUCKET: {
      if (_inner_resource->dirty_flag[type]) {
        int *label_array = (int *)_shared_buffer_array.get();
        memset(label_array, 0, sizeof(int) * static_cast<int>(_label_level));

        for (auto it = _labels.begin(); it != _labels.end(); ++it) {
          if (*it > static_cast<int>(_label_level) - 1) {
            std::stringstream ss;
            ss << "Input visible label : " << (int)(*it)
               << " is greater than the limit : "
               << static_cast<int>(_label_level) - 1 << " !";
            RENDERALGO_THROW_EXCEPTION(ss.str());
          }
          label_array[*it] = 1;
        }

        buffer->bind();
        buffer->load(static_cast<int>(_label_level) * sizeof(int), label_array,
                     GL_STATIC_DRAW);

        _inner_resource->dirty_flag[type] = false;
      }
      break;
    }

    case VISIBLE_LABEL_ARRAY: {
      if (_inner_resource->dirty_flag[type]) {
        int *label_array = (int *)_shared_buffer_array.get();
        memset(label_array, 0, sizeof(int) * static_cast<int>(_label_level));

        int idx = 0;
        for (auto it = _labels.begin(); it != _labels.end(); ++it, ++idx) {
          if (*it > static_cast<int>(_label_level) - 1) {
            std::stringstream ss;
            ss << "Input visible label : " << (int)(*it)
               << " is greater than the limit : "
               << static_cast<int>(_label_level) - 1 << " !";
            RENDERALGO_THROW_EXCEPTION(ss.str());
          }
          label_array[idx] = static_cast<int>(*it);
        }

        buffer->bind();
        buffer->load(idx * sizeof(int), label_array, GL_STATIC_DRAW);

        _inner_resource->dirty_flag[type] = false;
      }
      break;
    }

    case MASK_OVERLAY_COLOR_BUCKET: {
      if (_inner_resource->dirty_flag[type]) {
        float *color_array = (float *)_shared_buffer_array.get();
        memset(color_array, 0,
               sizeof(RGBAUnit) * static_cast<int>(_label_level));

        unsigned char label = 0;
        for (auto it = _mask_overlay_colors.begin();
             it != _mask_overlay_colors.end(); ++it) {
          label = it->first;
          if (label > static_cast<int>(_label_level) - 1) {
            std::stringstream ss;
            ss << "Input visible label : " << (int)(it->first)
               << " is greater than the limit : "
               << static_cast<int>(_label_level) - 1 << " !";
            RENDERALGO_THROW_EXCEPTION(ss.str());
          }
          color_array[label * 4] = it->second.r / 255.0f;
          color_array[label * 4 + 1] = it->second.g / 255.0f;
          color_array[label * 4 + 2] = it->second.b / 255.0f;
          color_array[label * 4 + 3] = it->second.a / 255.0f;
        }

        buffer->bind();
        buffer->load(static_cast<int>(_label_level) * sizeof(float) * 4,
                     color_array, GL_STATIC_DRAW);

        _inner_resource->dirty_flag[type] = false;
      }
      break;
    }

    case MATERIAL_BUCKET: {
      if (_inner_resource->dirty_flag[type]) {
        Material *material_array = (Material *)_shared_buffer_array.get();
        memset(material_array, 0,
               sizeof(Material) * static_cast<int>(_label_level));

        unsigned char label = 0;
        for (auto it = _material.begin(); it != _material.end(); ++it) {
          label = it->first;
          if (label > static_cast<int>(_label_level) - 1) {
            std::stringstream ss;
            ss << "Input visible label : " << (int)(it->first)
               << " is greater than the limit : "
               << static_cast<int>(_label_level) - 1 << " !";
            RENDERALGO_THROW_EXCEPTION(ss.str());
          }
          material_array[label] = it->second;
        }

        buffer->bind();
        buffer->load(static_cast<int>(_label_level) * sizeof(Material),
                     material_array, GL_STATIC_DRAW);

        _inner_resource->dirty_flag[type] = false;
      }
      break;
    }

    default: { RENDERALGO_THROW_EXCEPTION("Invalid buffer type!"); }
    }

    CHECK_GL_ERROR;

    return buffer;

  } catch (Exception &e) {
#ifdef _DEBUG
    std::cout << e.what();
#endif
    assert(false);
    throw e;
  }
}

void RayCasterInnerBuffer::set_mask_label_level(LabelLevel eLevel) {
  if (_label_level != eLevel) {
    _label_level = eLevel;
    memset(_inner_resource->dirty_flag, 1, sizeof(bool) * TYPE_END);
    _shared_buffer_array.reset(
        new char[static_cast<int>(_label_level) * sizeof(Material)]);
  }
}

void RayCasterInnerBuffer::set_window_level(float ww, float wl,
                                            unsigned char label) {
  const Vector2f wl_v2(ww, wl);
  auto it = _window_levels.find(label);
  if (it == _window_levels.end()) {
    _window_levels.insert(std::make_pair(label, wl_v2));
    _inner_resource->dirty_flag[WINDOW_LEVEL_BUCKET] = true;
  } else {
    if (wl_v2 != it->second) {
      it->second = wl_v2;
      _inner_resource->dirty_flag[WINDOW_LEVEL_BUCKET] = true;
    }
  }
}

void RayCasterInnerBuffer::set_visible_labels(
    std::vector<unsigned char> labels) {
  if (_labels != labels) {
    _labels = labels;
    _inner_resource->dirty_flag[VISIBLE_LABEL_BUCKET] = true;
    _inner_resource->dirty_flag[VISIBLE_LABEL_ARRAY] = true;
  }
}

const std::vector<unsigned char> &
RayCasterInnerBuffer::get_visible_labels() const {
  return _labels;
}

void RayCasterInnerBuffer::set_mask_overlay_color(
    std::map<unsigned char, RGBAUnit> colors) {
  if (_mask_overlay_colors != colors) {
    _mask_overlay_colors = colors;
    _inner_resource->dirty_flag[MASK_OVERLAY_COLOR_BUCKET] = true;
  }
}

void RayCasterInnerBuffer::set_mask_overlay_color(const RGBAUnit &color,
                                                  unsigned char label) {
  auto it = _mask_overlay_colors.find(label);
  if (it != _mask_overlay_colors.end()) {
    if (it->second != color) {
      it->second = color;
      _inner_resource->dirty_flag[MASK_OVERLAY_COLOR_BUCKET] = true;
    }
  } else {
    _mask_overlay_colors[label] = color;
    _inner_resource->dirty_flag[MASK_OVERLAY_COLOR_BUCKET] = true;
  }
}

const std::map<unsigned char, RGBAUnit> &
RayCasterInnerBuffer::get_mask_overlay_color() const {
  return _mask_overlay_colors;
}

void RayCasterInnerBuffer::set_material(const Material &matrial,
                                        unsigned char label) {
  auto it = _material.find(label);
  if (it == _material.end()) {
    _material.insert(std::make_pair(label, matrial));
    _inner_resource->dirty_flag[MATERIAL_BUCKET] = true;
  } else {
    if (matrial != it->second) {
      it->second = matrial;
      _inner_resource->dirty_flag[MATERIAL_BUCKET] = true;
    }
  }
}

MED_IMG_END_NAMESPACE
