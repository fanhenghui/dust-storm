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
    OpacityTransFunc(int width = 512);

    ~OpacityTransFunc();

    void set_width(int width);

    void add_point(float real_value, float a);

    void get_point_list(std::vector<OpacityTFPoint>& result_list);

    int get_width() const;

protected:

private:
    std::vector<OpacityTFPoint> _tp_points;
    std::vector<OpacityTFPoint> _result_points;
    int _width;
    bool _is_dirty;
};

MED_IMAGING_END_NAMESPACE
#endif