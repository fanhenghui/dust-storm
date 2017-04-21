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

    void calculate_brick_corner(std::shared_ptr<ImageData> image_data , unsigned int uiBrickSize , unsigned int uiBrickExpand , BrickCorner* pBrickCorner);

    void calculate_brick_unit( std::shared_ptr<ImageData> image_data , BrickCorner* pBrickCorner , unsigned int uiBrickSize , unsigned int uiBrickExpand , BrickUnit* pBrickUnit);

private:
    template<typename T>
    void calculate_brick_unit_i(BrickCorner& bc , BrickUnit& bu , std::shared_ptr<ImageData> image_data , unsigned int uiBrickSize , unsigned int uiBrickExpand);

    template<typename T>
    void calculate_brick_unit_kernel_i(unsigned int uiBegin , unsigned int uiEnd , BrickCorner* pBrickCorner , BrickUnit* pBrickUnit , std::shared_ptr<ImageData> image_data , unsigned int uiBrickSize , unsigned int uiBrickExpand);
};

MED_IMAGING_END_NAMESPACE

#endif