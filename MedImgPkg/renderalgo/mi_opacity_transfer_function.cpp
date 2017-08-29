#include "mi_opacity_transfer_function.h"
#include <cassert>
#include <cmath>

MED_IMG_BEGIN_NAMESPACE

OpacityTransFunc::OpacityTransFunc(int width /*= 512*/)
    : _width(width), _is_dirty(false) {}

OpacityTransFunc::~OpacityTransFunc() {}

void OpacityTransFunc::set_name(const std::string& lut_name) {
    _name = lut_name;
}

void OpacityTransFunc::set_width(int width) {
    if (width < 2) {
        RENDERALGO_THROW_EXCEPTION("invalid color transfer function width.");
    }

    _width = width;
    _is_dirty = true;
}

void OpacityTransFunc::add_point(float real_value, float a) {
    _tp_points.push_back(OpacityTFPoint(real_value, a));
    _is_dirty = true;
}

void OpacityTransFunc::get_point_list(
    std::vector<OpacityTFPoint>& result_list) {
    if (_is_dirty) { // Point changed
        if (_tp_points.empty()) {
            _result_points.clear();
            result_list = _result_points;
        }

        // Interpolation
        // 1 Sort the TFPoint from small to large
        // Bubble sort
        size_t tp_point_size = _tp_points.size();

        for (size_t i = 0; i < tp_point_size; ++i) {
            for (size_t j = 0; j < tp_point_size - 1 - i; ++j) {
                if (_tp_points[j].v > _tp_points[j + 1].v) {
                    OpacityTFPoint temp(_tp_points[j].v, _tp_points[j].a);
                    _tp_points[j] = _tp_points[j + 1];
                    _tp_points[j + 1] = temp;
                }
            }
        }

        // 2 Add to width count points
        // Expand point value to width , make the interpolation step is 1
        float max_value = _tp_points[tp_point_size - 1].v;
        float min_value = _tp_points[0].v;
        float expand_ratio =
            static_cast<float>(_width - 1) / (max_value - min_value);

        for (size_t i = 0; i < tp_point_size; ++i) {
            _tp_points[i].v = static_cast<float>(static_cast<int>(
                    (_tp_points[i].v - min_value) * expand_ratio + 0.5f));
        }

        // Interpolation
        _result_points.clear();
        _result_points.resize(_width);
        int idx = 0;

        for (size_t i = 0; i < tp_point_size - 1; ++i) {
            int gap = static_cast<int>(fabs(_tp_points[i + 1].v - _tp_points[i].v));

            if (0 == gap) {
                continue;
            }

            float step_alpha =
                (_tp_points[i + 1].a - _tp_points[i].a) / static_cast<float>(gap);
            float begin_alpha = _tp_points[i].a - step_alpha;
            float begin_value = _tp_points[i].v - 1.0f;

            for (int j = 0; j < gap; ++j) {
                begin_alpha += step_alpha;
                begin_value += 1.0f;
                _result_points[idx++] = (OpacityTFPoint(begin_value, begin_alpha));
            }
        }

        _result_points[idx] = (_tp_points[tp_point_size - 1]); // Add last one
        assert(idx == _width - 1);

        result_list = _result_points;
        _is_dirty = false;
    } else {
        result_list = _result_points;
    }
}

int OpacityTransFunc::get_width() const {
    return _width;
}

MED_IMG_END_NAMESPACE
