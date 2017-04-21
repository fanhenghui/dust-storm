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

    void set_width(int iWidth);

    void add_point(float fRealValue, float a);

    void get_point_list(std::vector<OpacityTFPoint>& vecResultList);

    int get_width() const;

protected:

private:
    std::vector<OpacityTFPoint> m_vecTFPoint;
    std::vector<OpacityTFPoint> m_vecResultList;
    int _width;
    bool m_bIsDataDirty;
};

MED_IMAGING_END_NAMESPACE
#endif