#ifndef MED_IMG_BRICK_GEOMETRY_H_
#define MED_IMG_BRICK_GEOMETRY_H_

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"
#include "boost/noncopyable.hpp"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export BrickGeometry : public boost::noncopyable
{
public:
    BrickGeometry();
    ~BrickGeometry();

    void set_parameter(
        const unsigned int(&volume_dim)[3] ,
        unsigned int brick_size ,
        unsigned int (&brick_dim)[3]);

    void calcualte();

    float* get_vertex_array();
    float* get_color_array();
    unsigned int* get_index_array();

protected:
private:
    struct BrickEleIndex
    {
        unsigned int idx[36];
    };

    std::unique_ptr<BrickEleIndex[]> _brick_idx_units;
    std::unique_ptr<float[]> _vertex_array;//Channel 3
    std::unique_ptr<float[]> _color_array;//Channel 4
    unsigned int _vertex_count;
};

MED_IMG_END_NAMESPACE