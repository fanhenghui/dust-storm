#ifndef MED_IMG_RENDERALGORITHM_MI_GPU_IMAGE_COMPRESSOR_H
#define MED_IMG_RENDERALGORITHM_MI_GPU_IMAGE_COMPRESSOR_H

#include <map>
#include <vector>

#include "renderalgo/mi_render_algo_export.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_gpu_resource_pair.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export GPUImgCompressor
{
public:
    GPUImgCompressor(GPUPlatform platform);
    ~GPUImgCompressor();
    //quality (0,100]
    int set_image(GPUCanvasPairPtr canvas, const std::vector<int>& qualitys);
    int resize_image(int width, int height);
    int compress(int quality, void* buffer, int& compress_size);
    float get_last_duration() const;

private:
    GPUPlatform _gpu_platform;
    struct InnerParams;
    std::map<int, InnerParams> _params;
    GPUCanvasPairPtr _canvas;
    float _duration;

private:
    DISALLOW_COPY_AND_ASSIGN(GPUImgCompressor);
};

MED_IMG_END_NAMESPACE


#endif
