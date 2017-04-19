#ifndef MED_IMAGING_RAY_CASTER_DEFINE_H_
#define MED_IMAGING_RAY_CASTER_DEFINE_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgCommon/mi_common_define.h"
#include "MedImgArithmetic/mi_color_unit.h"

MED_IMAGING_BEGIN_NAMESPACE

enum RayCastingStrategy
{
    CPU_BASE,
    CPU_BRICK_ACCELERATE,
    GPU_BASE,
};

enum LabelLevel
{
    L_8 = 8,
    L_16 = 16,
    L_24 = 24,
    L_32 = 32,
    L_40 = 40,
    L_48 = 48,
    L_56 = 56,
    L_64 = 64,
    L_72 = 72,
    L_80 = 80,
    L_88 = 88,
    L_96 = 96,
    L_104 = 104,
    L_112 = 112,
    L_120 = 120,
    L_128 = 128,
    L_256 = 256,
};

enum InterpolationMode
{
    NEARST = 0,
    LINEAR,
    CUBIC,
};

enum CompositeMode
{
    COMPOSITE_DVR = 0,
    COMPOSITE_MIP,
    COMPOSITE_MINIP,
    COMPOSITE_AVERAGE,
};

enum MaskMode
{
    MASK_NONE = 0,
    MASK_MULTI_LABEL,
};

enum ShadingMode
{
    SHADING_NONE = 0,
    SHADING_PHONG,
};

enum ColorInverseMode
{
    COLOR_INVERSE_DISABLE = 0, 
    COLOR_INVERSE_ENABLE,
};


static const RGBUnit kColorTransverse = RGBUnit((unsigned char)237 , (unsigned char)25, (unsigned char)35);//ºá¶ÏÃæ ºìÉ«
static const RGBUnit kColorCoronal = RGBUnit((unsigned char)255 , (unsigned char)128 , (unsigned char)0);//¹Ú×´Ãæ ½Û»ÆÉ«
static const RGBUnit kColorSagittal = RGBUnit((unsigned char)1 , (unsigned char)255, (unsigned char)64);//Ê¸×´Ãæ ÂÌÉ«

MED_IMAGING_END_NAMESPACE
#endif