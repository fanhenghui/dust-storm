#include "mi_opacity_transfer_function.h"

MED_IMAGING_BEGIN_NAMESPACE

OpacityTransFunc::OpacityTransFunc(int width /*= 512*/) :
_width(width), m_bIsDataDirty(false)
{}

OpacityTransFunc::~OpacityTransFunc()
{

}

void OpacityTransFunc::set_width(int width)
{
    _width = width;
    m_bIsDataDirty = true;
}

void OpacityTransFunc::add_point(float fRealValue, float a)
{
    m_vecTFPoint.push_back(OpacityTFPoint(fRealValue, a*255.0f));
    m_bIsDataDirty = true;
}

void OpacityTransFunc::get_point_list(std::vector<OpacityTFPoint>& vecResultList)
{
    if (m_bIsDataDirty)//Point changed
    {
        if (m_vecTFPoint.empty())
        {
            m_vecResultList.clear();
            vecResultList = m_vecResultList;
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
                    OpacityTFPoint temp(m_vecTFPoint[j].v, m_vecTFPoint[j].a);
                    m_vecTFPoint[j] = m_vecTFPoint[j + 1];
                    m_vecTFPoint[j + 1] = temp;
                }
            }
        }

        //2 Add to iWidht count points
        //Expand point value to width , make the interpolation step is 1
        float fMaxValue = m_vecTFPoint[uiTFPointSize - 1].v;
        float fMinValue = m_vecTFPoint[0].v;
        float fExpandRatio = static_cast<float>(_width - 1) / (fMaxValue - fMinValue);
        for (size_t i = 0; i < uiTFPointSize; ++i)
        {
            m_vecTFPoint[i].v = static_cast<float>(static_cast<int>((m_vecTFPoint[i].v - fMinValue)  * fExpandRatio + 0.5f));
        }

        //Interpolation
        m_vecResultList.clear();
        for (size_t i = 0; i < uiTFPointSize - 1; ++i)
        {
            int iGap = static_cast<int>(std::fabs(m_vecTFPoint[i + 1].v - m_vecTFPoint[i].v));
            if (0 == iGap)
            {
                continue;
            }
            float fStepAlpha = (m_vecTFPoint[i + 1].a - m_vecTFPoint[i].a) / static_cast<float>(iGap);
            float fBeginAlpha = m_vecTFPoint[i].a - fStepAlpha;
            float fBeginValue = m_vecTFPoint[i].v - 1.0f;
            for (int j = 0; j < iGap; ++j)
            {
                fBeginAlpha += fStepAlpha;
                fBeginValue += 1.0f;
                m_vecResultList.push_back(OpacityTFPoint(
                    fBeginValue,
                    fBeginAlpha));
            }
        }
        m_vecResultList.push_back(m_vecTFPoint[uiTFPointSize - 1]);//Add last one
        vecResultList = m_vecResultList;
        m_bIsDataDirty = false;
    }
    else
    {
        vecResultList = m_vecResultList;
    }
}

int OpacityTransFunc::get_width() const
{
    return _width;
}

MED_IMAGING_END_NAMESPACE
