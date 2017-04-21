#ifndef MED_IMAGING_BRICK_GENERATOR_H
#define MED_IMAGING_BRICK_GENERATOR_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class ImageData;
class RenderAlgo_Export BrickGenerator
{
public:
    BrickGenerator();

    ~BrickGenerator();

    void calculate_brick_corner(std::shared_ptr<ImageData> image_data , unsigned int brick_size , unsigned int brick_expand , BrickCorner* _brick_corner_array);

    void calculate_brick_unit( std::shared_ptr<ImageData> image_data , BrickCorner* _brick_corner_array , unsigned int brick_size , unsigned int brick_expand , BrickUnit* brick_unit_array);

private:
    template<typename T>
    void calculate_brick_unit_i(BrickCorner& bc , BrickUnit& bu , std::shared_ptr<ImageData> image_data , unsigned int brick_size , unsigned int brick_expand);

    template<typename T>
    void calculate_brick_unit_kernel_i(unsigned int begin , unsigned int end , BrickCorner* _brick_corner_array , BrickUnit* brick_unit_array , std::shared_ptr<ImageData> image_data , unsigned int brick_size , unsigned int brick_expand);
};

MED_IMAGING_END_NAMESPACE

#endif