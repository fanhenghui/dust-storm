#ifndef MED_IMG_BRICK_INFO_GENERATOR_H
#define MED_IMG_BRICK_INFO_GENERATOR_H

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export VolumeBrickInfoGenerator
{
public:
    VolumeBrickInfoGenerator();
    ~VolumeBrickInfoGenerator();
protected:
private:
};

class RenderAlgo_Export MaskBrickInfoGenerator
{
public:
    MaskBrickInfoGenerator();
    ~MaskBrickInfoGenerator();
protected:
private:
};

MED_IMG_END_NAMESPACE


#endif