#ifndef MEDIMGRENDERALGO_COLOR_TRNASFER_FUNCTION_H
#define MEDIMGRENDERALGO_COLOR_TRNASFER_FUNCTION_H

#include "renderalgo/mi_render_algo_export.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

enum ColorType { HSV, RGB };

struct ColorTFPoint {
    float v;
    float x;
    float y;
    float z;
    ColorTFPoint() : v(0), x(0), y(0), z(0) {}
    ColorTFPoint(float fv, float fx, float fy, float yz)
        : v(fv), x(fx), y(fy), z(yz) {}
    ColorTFPoint& operator=(const ColorTFPoint& right) {
        v = right.v;
        x = right.x;
        y = right.y;
        z = right.z;
        return *this;
    }
};

class RenderAlgo_Export ColorTransFunc {
public:
    ColorTransFunc(int width = 512);
    ~ColorTransFunc();

    void set_name(const std::string& lut_name);

    void set_width(int width);
    int get_width() const;

    void
    set_color_type(ColorType input_type,
                   ColorType interpolation_type); // output type is always rgb

    /// \0~255
    void add_rgb_point(float real_value, float r, float g, float b);

    /// \H (hue) 0~360
    // S(saturation) 0 ~ 1(0~100%)
    // V(value) 0~1(0~100%)
    void add_hsv_point(float real_value, float h, float s, float v);

    void get_point_list(std::vector<ColorTFPoint>& result_list);

    static ColorTFPoint hsv_to_rgb(const ColorTFPoint& hsv);
    static ColorTFPoint rgb_to_hsv(const ColorTFPoint& rgb);

protected:
private:
    std::string _name;
    std::vector<ColorTFPoint> _tp_points;
    std::vector<ColorTFPoint> _result_points;
    int _width;
    ColorType _interpolation_type;
    ColorType _input_type;
    bool _is_dirty;
};

MED_IMG_END_NAMESPACE
#endif