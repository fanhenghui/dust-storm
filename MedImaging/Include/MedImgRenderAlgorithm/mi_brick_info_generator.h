#ifndef MED_IMAGING_BRICK_INFO_GENERATOR_H
#define MED_IMAGING_BRICK_INFO_GENERATOR_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class ImageData;
class RenderAlgo_Export CPUVolumeBrickInfoGenerator
{
public:
    CPUVolumeBrickInfoGenerator();
    ~CPUVolumeBrickInfoGenerator();

    void calculate_brick_info(
        std::shared_ptr<ImageData> image_data , 
        unsigned int brick_size , 
        unsigned int brick_expand , 
        BrickCorner* _brick_corner_array , 
        BrickUnit* brick_unit_array , 
        VolumeBrickInfo* brick_info_array);
protected:
    template<typename T>
    void calculate_brick_info_i(
        BrickCorner& bc , 
        BrickUnit& bu ,
        VolumeBrickInfo& vbi,
        std::shared_ptr<ImageData> image_data , 
        unsigned int brick_size , 
        unsigned int brick_expand);

    template<typename T>
    void calculate_brick_info_kernel_i(
        unsigned int begin , 
        unsigned int end , 
        BrickCorner* _brick_corner_array , 
        BrickUnit* brick_unit_array , 
        VolumeBrickInfo* brick_info_array,
        std::shared_ptr<ImageData> image_data , 
        unsigned int brick_size , 
        unsigned int brick_expand);
private:
};

class RenderAlgo_Export GPUVolumeBrickInfoGenerator
{
public:
    GPUVolumeBrickInfoGenerator();
    ~GPUVolumeBrickInfoGenerator();
protected:
private:
};

class RenderAlgo_Export CPUMaskBrickInfoGenerator
{
public:
    CPUMaskBrickInfoGenerator();
    ~CPUMaskBrickInfoGenerator();
protected:
private:
};

class RenderAlgo_Export GPUMaskBrickInfoGenerator
{
public:
    GPUMaskBrickInfoGenerator();
    ~GPUMaskBrickInfoGenerator();
protected:
private:
};

MED_IMAGING_END_NAMESPACE


#endif