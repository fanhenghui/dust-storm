#include "mi_color_transfer_function.h"

MED_IMG_BEGIN_NAMESPACE

ColorTransFunc::ColorTransFunc(int width /*= 512 */) :_width(width), _interpolation_type(RGB), _input_type(RGB), _is_dirty(false)
{}

ColorTransFunc::~ColorTransFunc()
{

}

void ColorTransFunc::set_width(int width)
{
    _width = width;
    _is_dirty = true;
}

void ColorTransFunc::set_color_type(ColorType input_type, ColorType interpolation_type)
{
    _interpolation_type = interpolation_type;
    _input_type = input_type;
    _is_dirty = true;
}

void ColorTransFunc::add_rgb_point(float real_value, float x, float y, float z)
{
    //     if(x > 1.0f)
    //     {
    //         x /= 255.0f;
    //     }
    //     if(y> 1.0f)
    //     {
    //         y /= 255.0f;
    //     }
    //     if(z > 1.0f)
    //     {
    //         z /= 255.0f;
    //     }
    _tp_points.push_back(ColorTFPoint(real_value, x, y, z));
    _is_dirty = true;
}

void ColorTransFunc::add_hsv_point(float real_value, float x, float y, float z)
{
    _tp_points.push_back(ColorTFPoint(real_value, x, y, z));
    _is_dirty = true;
}

void ColorTransFunc::get_point_list(std::vector<ColorTFPoint>& result_list)
{
    if (_is_dirty)//Point changed
    {
        if (_tp_points.empty())
        {
            _result_points.clear();
            result_list = _result_points;
        }
        //Check input type and interpolation type
        if (_interpolation_type != _input_type)
        {
            if (RGB == _input_type)
            {
                for (auto it = _tp_points.begin(); it != _tp_points.end(); ++it)
                {
                    //Convert RGB tot HSV
                    *it = hsv_to_rgb(*it);
                }
            }
            if (HSV == _input_type)
            {
                for (auto it = _tp_points.begin(); it != _tp_points.end(); ++it)
                {
                    //Convert HSV tot RGB
                    *it = hsv_to_rgb(*it);
                }
            }
        }

        //Interpolation
        //1 Sort the TFPoint from small to large
        //Bubble sort
        size_t tp_point_size = _tp_points.size();
        for (size_t i = 0; i < tp_point_size; ++i)
        {
            for (size_t j = 0; j < tp_point_size - 1 - i; ++j)
            {
                if (_tp_points[j].v > _tp_points[j + 1].v)
                {
                    ColorTFPoint temp(_tp_points[j].v, _tp_points[j].x, _tp_points[j].y, _tp_points[j].z);
                    _tp_points[j] = _tp_points[j + 1];
                    _tp_points[j + 1] = temp;
                }
            }
        }

        //2 Add to iWidht count points
        //Expand point value to width , make the interpolation step is 1
        float max_value = _tp_points[tp_point_size - 1].v;
        float min_value = _tp_points[0].v;
        float expand_ratio = static_cast<float>(_width - 1) / (max_value - min_value);
        for (size_t i = 0; i < tp_point_size; ++i)
        {
            _tp_points[i].v = static_cast<float>(static_cast<int>((_tp_points[i].v - min_value)  * expand_ratio + 0.5f));
        }

        //Interpolation
        _result_points.clear();
        for (size_t i = 0; i < tp_point_size - 1; ++i)
        {
            int gap = static_cast<int>(fabs(_tp_points[i + 1].v - _tp_points[i].v));
            if (0 == gap)
            {
                continue;
            }
            float step_x = (_tp_points[i + 1].x - _tp_points[i].x) / static_cast<float>(gap);
            float step_y = (_tp_points[i + 1].y - _tp_points[i].y) / static_cast<float>(gap);
            float step_z = (_tp_points[i + 1].z - _tp_points[i].z) / static_cast<float>(gap);
            float begin_x = _tp_points[i].x - step_x;
            float begin_y = _tp_points[i].y - step_y;
            float begin_z = _tp_points[i].z - step_z;
            float begin_value = _tp_points[i].v - 1.0f;
            for (int j = 0; j < gap; ++j)
            {
                begin_x += step_x;
                begin_y += step_y;
                begin_z += step_z;
                begin_value += 1.0f;
                _result_points.push_back(ColorTFPoint(
                    begin_value,
                    begin_x,
                    begin_y,
                    begin_z));
            }
        }
        _result_points.push_back(_tp_points[tp_point_size - 1]);//Add last one

        //Transfer HSV interpolation to RGB
        if (HSV == _interpolation_type)
        {
            for (size_t i = 0; i < _result_points.size(); ++i)
            {
                _result_points[i] = hsv_to_rgb(_result_points[i]);
            }
        }

        result_list = _result_points;
        _is_dirty = false;
    }
    else
    {
        result_list = _result_points;
    }
}

int ColorTransFunc::get_width() const
{
    return _width;
}

ColorTFPoint ColorTransFunc::hsv_to_rgb(const ColorTFPoint& hsv)
{
    /// \输入的HSV的范围是 H（hue）0~359
    //S(saturation) 0 ~ 1(0~100%)
    //V(value) 0~1(0~100%)
    float h = hsv.x / 359.0f;
    float s = hsv.y;
    float v = hsv.z;

    float r, g, b;

    const float onethird = 1.0f / 3.0f;
    const float onesixth = 1.0f / 6.0f;
    const float twothird = 2.0f / 3.0f;
    const float fivesixth = 5.0f / 6.0f;

    // compute RGB from HSV
    if (h > onesixth && h <= onethird) // green/red
    {
        g = 1.0f;
        r = (onethird - h) / onesixth;
        b = 0.0f;
    }
    else if (h > onethird && h <= 0.5f) // green/blue
    {
        g = 1.0f;
        b = (h - onethird) / onesixth;
        r = 0.0f;
    }
    else if (h > 0.5f && h <= twothird) // blue/green
    {
        b = 1.0f;
        g = (twothird - h) / onesixth;
        r = 0.0f;
    }
    else if (h > twothird && h <= fivesixth) // blue/red
    {
        b = 1.0f;
        r = (h - twothird) / onesixth;
        g = 0.0f;
    }
    else if (h > fivesixth && h <= 1.0f) // red/blue
    {
        r = 1.0f;
        b = (1.0f - h) / onesixth;
        g = 0.0f;
    }
    else // red/green
    {
        r = 1.0f;
        g = h / onesixth;
        b = 0.0f;
    }

    // add Saturation to the equation.
    r = (s * r + (1.0f - s));
    g = (s * g + (1.0f - s));
    b = (s * b + (1.0f - s));

    r *= v*255.0f;
    g *= v*255.0f;
    b *= v*255.0f;

    return ColorTFPoint(hsv.v, r, g, b);
}

ColorTFPoint ColorTransFunc::rgb_to_hsv(const ColorTFPoint& rgb)
{
    /// \输入的RGB的范围是 0~1
    float onethird = 1.0f / 3.0f;
    float onesixth = 1.0f / 6.0f;
    float twothird = 2.0f / 3.0f;

    float cmax, cmin;
    float r = rgb.x / 255.0f;
    float g = rgb.y / 255.0f;
    float b = rgb.z / 255.0f;
    float h, s, v;

    cmax = r;
    cmin = r;
    if (g > cmax)
    {
        cmax = g;
    }
    else if (g < cmin)
    {
        cmin = g;
    }
    if (b > cmax)
    {
        cmax = b;
    }
    else if (b < cmin)
    {
        cmin = b;
    }
    v = cmax;

    if (v > 0.0f)
    {
        s = (cmax - cmin) / cmax;
    }
    else
    {
        s = 0.0f;
    }
    if (s > 0.0f)
    {
        if (r == cmax)
        {
            h = onesixth * (g - b) / (cmax - cmin);
        }
        else if (g == cmax)
        {
            h = onethird + onesixth * (b - r) / (cmax - cmin);
        }
        else
        {
            h = twothird + onesixth * (r - g) / (cmax - cmin);
        }
        if (h < 0.0f)
        {
            h += 1.0f;
        }
    }
    else
    {
        h = 0.0f;
    }
    return ColorTFPoint(rgb.v, h*359.0f, s, v);
}

MED_IMG_END_NAMESPACE