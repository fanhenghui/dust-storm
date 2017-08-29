#include "mi_ray_casting_cpu.h"

#include "boost/thread.hpp"

#include "arithmetic/mi_sampler.h"
#include "glresource/mi_gl_utils.h"
#include "io/mi_image_data.h"

#include "mi_entry_exit_points.h"
#include "mi_ray_caster.h"
#include "mi_ray_caster_canvas.h"

MED_IMG_BEGIN_NAMESPACE

RayCastingCPU::RayCastingCPU(std::shared_ptr<RayCaster> ray_caster)
    : _ray_caster(ray_caster), _width(32), _height(32), _entry_points(nullptr),
      _exit_points(nullptr), _volume_data_array(nullptr),
      _mask_data_array(nullptr), _canvas_array(nullptr) {
    _dim[0] = _dim[1] = _dim[2] = 32;
}

RayCastingCPU::~RayCastingCPU() {}

void RayCastingCPU::render(int test_code) {
    try {
        std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster);

        // Volume info
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster->_entry_exit_points);
        ray_caster->_entry_exit_points->get_display_size(_width, _height);

        std::shared_ptr<ImageData> volume_data = ray_caster->_volume_data;
        RENDERALGO_CHECK_NULL_EXCEPTION(volume_data);
        memcpy(_dim, volume_data->_dim, sizeof(unsigned int) * 3);
        _volume_data_array = volume_data->get_pixel_pointer();

        std::shared_ptr<ImageData> mask_data = ray_caster->_mask_data;
        _mask_data_array = (unsigned char*)(mask_data->get_pixel_pointer());

        // Entry exit points
        _entry_points = ray_caster->_entry_exit_points->get_entry_points_array();
        _exit_points = ray_caster->_entry_exit_points->get_exit_points_array();

        // Canvas
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster->_canvas);
        _canvas_array = ray_caster->_canvas->get_color_array();
        RENDERALGO_CHECK_NULL_EXCEPTION(_canvas_array);

        //////////////////////////////////////////////////////////////////////////
        // For testing entry & exit points
        if (0 != test_code) {
            render_entry_exit_points_i(test_code);
            ray_caster->_canvas->update_color_array();
            return;
        }

        //////////////////////////////////////////////////////////////////////////

        switch (volume_data->_data_type) {
        case USHORT: {
            ray_casting_i<unsigned short>(ray_caster);
            break;
        }

        case SHORT: {
            ray_casting_i<short>(ray_caster);
            break;
        }

        case FLOAT: {
            ray_casting_i<float>(ray_caster);
            break;
        }

        default:
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }

        // mask label overlay
        if (ray_caster->_mask_overlay_mode == MASK_OVERLAY_ENABLE) {
            mask_overlay_i(ray_caster);
        }

        CHECK_GL_ERROR;

        if (COMPOSITE_AVERAGE == ray_caster->_composite_mode ||
                COMPOSITE_MIP == ray_caster->_composite_mode ||
                COMPOSITE_MINIP == ray_caster->_composite_mode) {
            ray_caster->_canvas->update_color_array();
        }

        CHECK_GL_ERROR;

    } catch (const Exception& e) {
        //#ifdef _DEBUG
        // TODO LOG
        std::cout << e.what();
        //#endif
        assert(false);
        throw e;
    } catch (const std::exception& e) {
#ifdef _DEBUG
        // TODO LOG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
}

template <class T>
void RayCastingCPU::ray_casting_i(std::shared_ptr<RayCaster> ray_caster) {
    switch (ray_caster->_composite_mode) {
    case COMPOSITE_AVERAGE: {
        ray_casting_average_i<T>(ray_caster);
        break;
    }

    case COMPOSITE_MIP: {
        ray_casting_mip_i<T>(ray_caster);
        break;
    }

    case COMPOSITE_MINIP: {
        ray_casting_minip_i<T>(ray_caster);
        break;
    }

    default:
        break;
    }
}

template <class T>
void RayCastingCPU::ray_casting_average_i(
    std::shared_ptr<RayCaster> ray_caster) {
    const int pixel_sum = _width * _height;

#ifndef _DEBUG
    #pragma omp parallel for
#endif

    for (int idx = 0; idx < pixel_sum; ++idx) {
        const Sampler<T> sampler;

        const int y = idx / _width;
        const int x = idx - y * _width;

        // 1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] <
                          -0.5f; // -1.0 for skip , 0  for valid entry exit points

        if (skip) {
            _canvas_array[idx] = RGBAUnit();
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize() * ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step = (int)step_float;

        if (step == 0) { //��֤���ٻ���һ��
            step = 1;
        }

        // 2 Integrate
        const float ratio = 1000.0f;
        const float ratio_r = 1.0f / 1000.0f;
        float sum = 0.0f;
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;

        for (int i = 0; i < step; ++i) {
            sample_pos += (dir_step * float(i));

            sample_value = sampler.sample_3d_linear(
                               sample_pos._m[0], sample_pos._m[1], sample_pos._m[2], _dim[0],
                               _dim[1], _dim[2], (T*)_volume_data_array);

            sum += sample_value * ratio_r;
        }

        const float result_gray = sum * (1.0f / step) * ratio;

        // 3Apply window level
        const float min_wl_gray =
            ray_caster->_global_wl - ray_caster->_global_ww * 0.5f;
        const float gray = (result_gray - min_wl_gray) / ray_caster->_global_ww;

        // 4Apply pseudo color
        // TODO just gray
        _canvas_array[idx] = RGBAUnit::norm_to_integer(gray, gray, gray);
    }
}

template <class T>
void RayCastingCPU::ray_casting_mip_i(std::shared_ptr<RayCaster> ray_caster) {
    const int pixel_sum = _width * _height;

#ifndef _DEBUG
    #pragma omp parallel for
#endif

    for (int idx = 0; idx < pixel_sum; ++idx) {
        const Sampler<T> sampler;

        const int y = idx / _width;
        const int x = idx - y * _width;

        // 1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] <
                          -0.5f; // -1.0 for skip , 0  for valid entry exit points

        if (skip) {
            _canvas_array[idx] = RGBAUnit();
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize() * ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step = (int)step_float;

        if (step == 0) { //��֤���ٻ���һ��
            step = 1;
        }

        // 2 Integrate
        float max_gray = -65535.0f;
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;

        for (int i = 0; i < step; ++i) {
            sample_pos += (dir_step * float(i));

            sample_value = sampler.sample_3d_linear(
                               sample_pos._m[0], sample_pos._m[1], sample_pos._m[2], _dim[0],
                               _dim[1], _dim[2], (T*)_volume_data_array);

            max_gray = sample_value > max_gray ? sample_value : max_gray;
        }

        // 3Apply window level
        const float min_wl_gray =
            ray_caster->_global_wl - ray_caster->_global_ww * 0.5f;
        const float gray = (max_gray - min_wl_gray) / ray_caster->_global_ww;

        // 4Apply pseudo color
        // TODO just gray
        _canvas_array[idx] = RGBAUnit::norm_to_integer(gray, gray, gray);
    }
}

template <class T>
void RayCastingCPU::ray_casting_minip_i(std::shared_ptr<RayCaster> ray_caster) {
    const int pixel_sum = _width * _height;

#ifndef _DEBUG
    #pragma omp parallel for
#endif

    for (int idx = 0; idx < pixel_sum; ++idx) {
        const Sampler<T> sampler;

        const int y = idx / _width;
        const int x = idx - y * _width;

        // 1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] <
                          -0.5f; // -1.0 for skip , 0  for valid entry exit points

        if (skip) {
            _canvas_array[idx] = RGBAUnit();
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize() * ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step = (int)step_float;

        if (step == 0) { //��֤���ٻ���һ��
            step = 1;
        }

        // 2 Integrate
        float max_gray = -65535.0f;
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;

        for (int i = 0; i < step; ++i) {
            sample_pos += (dir_step * float(i));

            sample_value = sampler.sample_3d_linear(
                               sample_pos._m[0], sample_pos._m[1], sample_pos._m[2], _dim[0],
                               _dim[1], _dim[2], (T*)_volume_data_array);

            max_gray = sample_value > max_gray ? sample_value : max_gray;
        }

        // 3Apply window level
        const float min_wl_gray =
            ray_caster->_global_wl - ray_caster->_global_ww * 0.5f;
        const float gray = (max_gray - min_wl_gray) / ray_caster->_global_ww;

        // 4Apply pseudo color
        // TODO just gray
        _canvas_array[idx] = RGBAUnit::norm_to_integer(gray, gray, gray);
    }
}

void RayCastingCPU::render_entry_exit_points_i(int test_code) {
    Vector3f vDimR(1.0f / _dim[0], 1.0f / _dim[1], 1.0f / _dim[2]);
    const int pixel_sum = _width * _height;

    if (1 == test_code) {
        for (int i = 0; i < pixel_sum; ++i) {
            Vector3f start_point(_entry_points[i]._m128);
            start_point = start_point * vDimR;
            _canvas_array[i] = RGBAUnit::norm_to_integer(
                                   start_point._m[0], start_point._m[1], start_point._m[2]);
        }
    } else {
        for (int i = 0; i < pixel_sum; ++i) {
            Vector3f end_point(_exit_points[i]._m128);
            end_point = end_point * vDimR;
            _canvas_array[i] = RGBAUnit::norm_to_integer(
                                   end_point._m[0], end_point._m[1], end_point._m[2]);
        }
    }
}

namespace {
// Encoding label to intger array 4*32 can contain 0~127 labels
void label_encode(unsigned char label, int (&mask_flag)[4]) {
    if (label < 32) {
        mask_flag[0] = mask_flag[0] | (1 << label);
    } else if (label < 64) {
        mask_flag[1] = mask_flag[1] | (1 << (label - 32));
    } else if (label < 96) {
        mask_flag[2] = mask_flag[2] | (1 << (label - 64));
    } else {
        mask_flag[3] = mask_flag[3] | (1 << (label - 96));
    }
}

// Decoding label from intger array 4*32 can contain 0~127 labels
bool label_decode(unsigned char label, const int (&mask_flag)[4]) {

    bool is_hitted = false;

    if (label < 32) {
        is_hitted = ((1 << label) & mask_flag[0]) != 0;
    } else if (label < 64) {
        is_hitted = ((1 << (label - 32)) & mask_flag[1]) != 0;
    } else if (label < 96) {
        is_hitted = ((1 << (label - 64)) & mask_flag[2]) != 0;
    } else {
        is_hitted = ((1 << (label - 96)) & mask_flag[3]) != 0;
    }

    return is_hitted;
}
}

void RayCastingCPU::mask_overlay_i(std::shared_ptr<RayCaster> ray_caster) {
    const int pixel_sum = _width * _height;

    const std::vector<unsigned char>& visible_labels =
        ray_caster->get_visible_labels();
    const std::map<unsigned char, RGBAUnit>& mask_overlay_color =
        ray_caster->get_mask_overlay_color();

#ifndef _DEBUG
    #pragma omp parallel for
#endif

    for (int idx = 0; idx < pixel_sum; ++idx) {
        Sampler<unsigned char> sampler;
        int active_mask_code[4] = {0, 0, 0, 0};

        const int y = idx / _width;
        const int x = idx - y * _width;

        // 1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] <
                          -0.5f; // -1.0 for skip , 0  for valid entry exit points

        if (skip) {
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize() * ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step = (int)step_float;

        if (step == 0) { //��֤���ٻ���һ��
            step = 1;
        }

        // 2 Integrate
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;
        unsigned char label = 0;

        for (int i = 0; i < step; ++i) {
            sample_pos += (dir_step * float(i));

            sample_value = sampler.sample_3d_nearst(
                               sample_pos._m[0], sample_pos._m[1], sample_pos._m[2], _dim[0],
                               _dim[1], _dim[2], _mask_data_array);

            label = static_cast<unsigned char>(sample_value);

            if (0 != label) {
                label_encode(label, active_mask_code);
            }
        }

        // 3 Apply mask overlay color
        Vector3f current_color(0, 0, 0);
        float current_alpha = 0;

        for (auto it = visible_labels.begin(); it != visible_labels.end(); ++it) {
            unsigned char vis_label = *it;

            if (label_decode(vis_label, active_mask_code)) {
                auto it = mask_overlay_color.find(vis_label);

                if (it != mask_overlay_color.end()) {
                    Vector3f label_color((float)it->second.r, (float)it->second.g,
                                         (float)it->second.b);
                    float label_alpha = (float)it->second.a / 255.0f;

                    current_color += label_color * ((1 - current_alpha) * label_alpha);
                    current_alpha += (1 - current_alpha) * label_alpha;
                }
            }
        }

        // 4 Blend
        RGBAUnit& previous_rgba = _canvas_array[idx];
        current_color = Vector3f((float)previous_rgba.r, (float)previous_rgba.g,
                                 (float)previous_rgba.b) +
                        current_color * current_alpha;

        RGBAUnit current_rgba;
        current_rgba.r = current_color._m[0] > 255.0f
                         ? 255
                         : static_cast<unsigned char>(current_color._m[0]);
        current_rgba.g = current_color._m[1] > 255.0f
                         ? 255
                         : static_cast<unsigned char>(current_color._m[1]);
        current_rgba.b = current_color._m[2] > 255.0f
                         ? 255
                         : static_cast<unsigned char>(current_color._m[2]);
        current_rgba.a = 255;
        _canvas_array[idx] = current_rgba;
    }
}

MED_IMG_END_NAMESPACE