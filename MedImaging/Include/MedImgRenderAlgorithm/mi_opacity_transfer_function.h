#ifndef OPACITY_TRNASFER_FUNCTION_H_
#define OPACITY_TRNASFER_FUNCTION_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

struct OpacityTFPoint
{
    float v;
    float a;
    OpacityTFPoint():v(0),a(0)
    {}
    OpacityTFPoint(float fv, float fa) :
    v(fv), a(fa)
    {}
    OpacityTFPoint& operator=(const OpacityTFPoint& right)
    {
        v = right.v;
        a = right.a;
        return *this;
    }
};

class RenderAlgo_Export OpacityTransFunc
{
public:
    OpacityTransFunc(int iWidth = 512);

    ~OpacityTransFunc();

    void SetWidth(int iWidth);

    void AddPoint(float fRealValue, float a);

    void GetPointList(std::vector<OpacityTFPoint>& vecResultList);

    int GetWidth() const;

protected:

private:
    std::vector<OpacityTFPoint> m_vecTFPoint;
    std::vector<OpacityTFPoint> m_vecResultList;
    int m_iWidth;
    bool m_bIsDataDirty;
};

MED_IMAGING_END_NAMESPACE
#endif