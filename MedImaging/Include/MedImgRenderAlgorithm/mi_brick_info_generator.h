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
        std::shared_ptr<ImageData> pImgData , 
        unsigned int uiBrickSize , 
        unsigned int uiBrickExpand , 
        BrickCorner* pBrickCorner , 
        BrickUnit* pBrickUnit , 
        VolumeBrickInfo* pBrickInfo);
protected:
    template<typename T>
    void calculate_brick_info_i(
        BrickCorner& bc , 
        BrickUnit& bu ,
        VolumeBrickInfo& vbi,
        std::shared_ptr<ImageData> pImgData , 
        unsigned int uiBrickSize , 
        unsigned int uiBrickExpand);

    template<typename T>
    void calculate_brick_info_kernel_i(
        unsigned int uiBegin , 
        unsigned int uiEnd , 
        BrickCorner* pBrickCorner , 
        BrickUnit* pBrickUnit , 
        VolumeBrickInfo* pBrickInfo,
        std::shared_ptr<ImageData> pImgData , 
        unsigned int uiBrickSize , 
        unsigned int uiBrickExpand);
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