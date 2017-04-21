#ifndef COLOR_TRNASFER_FUNCTION_H_
#define COLOR_TRNASFER_FUNCTION_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

enum ColorType
{
    HSV,
    RGB
};

struct ColorTFPoint
{
    float v;
    float x;
    float y;
    float z;
    ColorTFPoint():v(0),x(0),y(0),z(0)
    {}
    ColorTFPoint(float fv, float fx, float fy, float yz) :
    v(fv), x(fx), y(fy), z(yz)
    {}
    ColorTFPoint& operator=(const ColorTFPoint& right)
    {
        v = right.v;
        x = right.x;
        y = right.y;
        z = right.z;
        return *this;
    }
};

class RenderAlgo_Export ColorTransFunc
{
public:
    ColorTransFunc(int width = 512);

    ~ColorTransFunc();

    void set_width(int width);

    void set_color_type(ColorType eInputColorType, ColorType eInterpolationColorType);

    /// \输入的RGB的范围是 0~255
    void add_rgb_point(float fRealValue, float x, float y, float z);

    /// \输入的HSV的范围是 H（hue）0~359
    //S(saturation) 0 ~ 1(0~100%)
    //V(value) 0~1(0~100%)
    void add_hsv_point(float fRealValue, float x, float y, float z);

    void get_point_list(std::vector<ColorTFPoint>& vecResultList);

    int get_width() const;

    static ColorTFPoint hsv_to_rgb(const ColorTFPoint& hsv);

    static ColorTFPoint rgb_to_hsv(const ColorTFPoint& rgb);

protected:

private:
    std::vector<ColorTFPoint> m_vecTFPoint;
    std::vector<ColorTFPoint> m_vecResultList;
    int _width;
    ColorType m_eInterpolationColorType;
    ColorType m_eInputColorType;
    bool m_bIsDataDirty;
};

MED_IMAGING_END_NAMESPACE
#endif