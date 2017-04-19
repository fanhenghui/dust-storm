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
    ColorTransFunc(int iWidth = 512);

    ~ColorTransFunc();

    void SetWidth(int iWidth);

    void SetColorType(ColorType eInputColorType, ColorType eInterpolationColorType);

    /// \输入的RGB的范围是 0~255
    void AddRGBPoint(float fRealValue, float x, float y, float z);

    /// \输入的HSV的范围是 H（hue）0~359
    //S(saturation) 0 ~ 1(0~100%)
    //V(value) 0~1(0~100%)
    void AddHSVPoint(float fRealValue, float x, float y, float z);

    void GetPointList(std::vector<ColorTFPoint>& vecResultList);

    int GetWidth() const;

    static ColorTFPoint HSV2RGB(const ColorTFPoint& hsv);

    static ColorTFPoint RGB2HSV(const ColorTFPoint& rgb);

protected:

private:
    std::vector<ColorTFPoint> m_vecTFPoint;
    std::vector<ColorTFPoint> m_vecResultList;
    int m_iWidth;
    ColorType m_eInterpolationColorType;
    ColorType m_eInputColorType;
    bool m_bIsDataDirty;
};

MED_IMAGING_END_NAMESPACE
#endif