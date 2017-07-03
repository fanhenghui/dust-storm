#ifndef MEDCIAL_IMAGING_BRICK_UTILS_H
#define MEDCIAL_IMAGING_BRICK_UTILS_H

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"

#include "boost/thread/mutex.hpp"

#include "MedImgArithmetic/mi_vector3f.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export BrickUtils
{
public:
    static BrickUtils* instance();

    ~BrickUtils();

    void set_brick_size(unsigned int size);

    unsigned int GetBrickSize();

    void set_brick_expand(unsigned int size);

    unsigned int get_brick_expand();

    void get_brick_dim(const unsigned int(&volume_dim)[3] , unsigned int(&brick_dim)[3] , unsigned int brick_size);

private:
    BrickUtils();

private:
    static BrickUtils* _s_instance;
    static boost::mutex _s_mutex;

    unsigned int _brick_size;
    unsigned int _brick_expand;

};

MED_IMG_END_NAMESPACE

#endif