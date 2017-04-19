#include "mi_color_transfer_function.h"

MED_IMAGING_BEGIN_NAMESPACE

ColorTransFunc::ColorTransFunc(int iWidth /*= 512 */) :
m_iWidth(iWidth), m_eInterpolationColorType(RGB), m_eInputColorType(RGB), m_bIsDataDirty(false)
{}

ColorTransFunc::~ColorTransFunc()
{

}

void ColorTransFunc::SetWidth(int iWidth)
{
    m_iWidth = iWidth;
    m_bIsDataDirty = true;
}

void ColorTransFunc::SetColorType(ColorType eInputColorType, ColorType eInterpolationColorType)
{
    m_eInterpolationColorType = eInterpolationColorType;
    m_eInputColorType = eInputColorType;
    m_bIsDataDirty = true;
}

void ColorTransFunc::AddRGBPoint(float fRealValue, float x, float y, float z)
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
    m_vecTFPoint.push_back(ColorTFPoint(fRealValue, x, y, z));
    m_bIsDataDirty = true;
}

void ColorTransFunc::AddHSVPoint(float fRealValue, float x, float y, float z)
{
    m_vecTFPoint.push_back(ColorTFPoint(fRealValue, x, y, z));
    m_bIsDataDirty = true;
}

void ColorTransFunc::GetPointList(std::vector<ColorTFPoint>& vecResultList)
{
    if (m_bIsDataDirty)//Point changed
    {
        if (m_vecTFPoint.empty())
        {
            m_vecResultList.clear();
            vecResultList = m_vecResultList;
        }
        //Check input type and interpolation type
        if (m_eInterpolationColorType != m_eInputColorType)
        {
            if (RGB == m_eInputColorType)
            {
                for (auto it = m_vecTFPoint.begin(); it != m_vecTFPoint.end(); ++it)
                {
                    //Convert RGB tot HSV
                    *it = RGB2HSV(*it);
                }
            }
            if (HSV == m_eInputColorType)
            {
                for (auto it = m_vecTFPoint.begin(); it != m_vecTFPoint.end(); ++it)
                {
                    //Convert HSV tot RGB
                    *it = HSV2RGB(*it);
                }
            }
        }

        //Interpolation
        //1 Sort the TFPoint from small to large
        //Bubble sort
        size_t uiTFPointSize = m_vecTFPoint.size();
        for (size_t i = 0; i < uiTFPointSize; ++i)
        {
            for (size_t j = 0; j < uiTFPointSize - 1 - i; ++j)
            {
                if (m_vecTFPoint[j].v > m_vecTFPoint[j + 1].v)
                {
                    ColorTFPoint temp(m_vecTFPoint[j].v, m_vecTFPoint[j].x, m_vecTFPoint[j].y, m_vecTFPoint[j].z);
                    m_vecTFPoint[j] = m_vecTFPoint[j + 1];
                    m_vecTFPoint[j + 1] = temp;
                }
            }
        }

        //2 Add to iWidht count points
        //Expand point value to iWidth , make the interpolation step is 1
        float fMaxValue = m_vecTFPoint[uiTFPointSize - 1].v;
        float fMinValue = m_vecTFPoint[0].v;
        float fExpandRatio = static_cast<float>(m_iWidth - 1) / (fMaxValue - fMinValue);
        for (size_t i = 0; i < uiTFPointSize; ++i)
        {
            m_vecTFPoint[i].v = static_cast<float>(static_cast<int>((m_vecTFPoint[i].v - fMinValue)  * fExpandRatio + 0.5f));
        }

        //Interpolation
        m_vecResultList.clear();
        for (size_t i = 0; i < uiTFPointSize - 1; ++i)
        {
            int iGap = static_cast<int>(fabs(m_vecTFPoint[i + 1].v - m_vecTFPoint[i].v));
            if (0 == iGap)
            {
                continue;
            }
            float fStepX = (m_vecTFPoint[i + 1].x - m_vecTFPoint[i].x) / static_cast<float>(iGap);
            float fStepY = (m_vecTFPoint[i + 1].y - m_vecTFPoint[i].y) / static_cast<float>(iGap);
            float fStepZ = (m_vecTFPoint[i + 1].z - m_vecTFPoint[i].z) / static_cast<float>(iGap);
            float fBeginX = m_vecTFPoint[i].x - fStepX;
            float fBeginY = m_vecTFPoint[i].y - fStepY;
            float fBeginZ = m_vecTFPoint[i].z - fStepZ;
            float fBeginValue = m_vecTFPoint[i].v - 1.0f;
            for (int j = 0; j < iGap; ++j)
            {
                fBeginX += fStepX;
                fBeginY += fStepY;
                fBeginZ += fStepZ;
                fBeginValue += 1.0f;
                m_vecResultList.push_back(ColorTFPoint(
                    fBeginValue,
                    fBeginX,
                    fBeginY,
                    fBeginZ));
            }
        }
        m_vecResultList.push_back(m_vecTFPoint[uiTFPointSize - 1]);//Add last one

        //Transfer HSV interpolation to RGB
        if (HSV == m_eInterpolationColorType)
        {
            for (size_t i = 0; i < m_vecResultList.size(); ++i)
            {
                m_vecResultList[i] = HSV2RGB(m_vecResultList[i]);
            }
        }

        vecResultList = m_vecResultList;
        m_bIsDataDirty = false;
    }
    else
    {
        vecResultList = m_vecResultList;
    }
}

int ColorTransFunc::GetWidth() const
{
    return m_iWidth;
}

ColorTFPoint ColorTransFunc::HSV2RGB(const ColorTFPoint& hsv)
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

ColorTFPoint ColorTransFunc::RGB2HSV(const ColorTFPoint& rgb)
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

MED_IMAGING_END_NAMESPACE