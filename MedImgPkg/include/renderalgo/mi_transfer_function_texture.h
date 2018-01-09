#ifndef MED_IMG_RENDERALGORITHM_MI_TRANSFER_FUNCTION_TEXTURE_H
#define MED_IMG_RENDERALGORITHM_MI_TRANSFER_FUNCTION_TEXTURE_H

#include "renderalgo/mi_entry_exit_points.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_gpu_resource_pair.h"

MED_IMG_BEGIN_NAMESPACE

class ColorTransFunc;
class OpacityTransFunc;
class RenderAlgo_Export TransferFunctionTexture
{
public:
    TransferFunctionTexture(RayCastingStrategy strategy, GPUPlatform platform);
    ~TransferFunctionTexture();
    
    void initialize(LabelLevel label_level);

    void set_pseudo_color(std::shared_ptr<ColorTransFunc> color);
    void set_color_opacity(std::shared_ptr<ColorTransFunc> color,
        std::shared_ptr<OpacityTransFunc> opacity, unsigned char label);
    
    GPUTexture1DPairPtr get_pseudo_color_texture();
    GPUTexture1DArrayPairPtr get_color_opacity_texture_array();

private:
    RayCastingStrategy _strategy;
    GPUPlatform _gpu_platform;
    LabelLevel _label_level;
    bool _init;
    GPUTexture1DPairPtr _pseudo_color_texture;
    GPUTexture1DArrayPairPtr _color_opacity_texture_array;
    GLResourceShield _res_shield;
private:
    DISALLOW_COPY_AND_ASSIGN(TransferFunctionTexture);
};


MED_IMG_END_NAMESPACE

#endif
