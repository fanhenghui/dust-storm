#ifndef MEDCIAL_IMAGING_BRICK_UTILS_H
#define MEDCIAL_IMAGING_BRICK_UTILS_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"

#include "boost/thread/mutex.hpp"

#include "MedImgArithmetic/mi_vector3f.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export BrickUtils
{
public:
    static BrickUtils* instance();

    ~BrickUtils();

    void set_brick_size(unsigned int uiSize);

    unsigned int GetBrickSize();

    void set_brick_expand(unsigned int uiSize);

    unsigned int get_brick_expand();

    void get_brick_dim(const unsigned int(&uiVolumeDim)[3] , unsigned int(&uiBrickDim)[3] , unsigned int uiBrickSize);

private:
    BrickUtils();

private:
    static BrickUtils* m_instance;
    static boost::mutex m_mutex;

    unsigned int m_uiBrickSize;
    unsigned int m_uiBrickExpand;

};

MED_IMAGING_END_NAMESPACE

#endif