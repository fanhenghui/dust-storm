#ifndef MED_IMG_RENDERALGORITHM_MI_GPU_IMAGE_COMPRESSOR_H
#define MED_IMG_RENDERALGORITHM_MI_GPU_IMAGE_COMPRESSOR_H

#include "renderalgo/mi_render_algo_export.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export GPUImgCompressor
{
public:
    GPUImgCompressor();
    ~GPUImgCompressor();
    void compress();
protected:
private:
};

MED_IMG_END_NAMESPACE


#endif
