#ifndef MED_IMG_RAY_CASTER_DEFINE_H_
#define MED_IMG_RAY_CASTER_DEFINE_H_

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgIO/mi_io_define.h"
#include "MedImgArithmetic/mi_color_unit.h"

MED_IMG_BEGIN_NAMESPACE

enum RayCastingStrategy
{
    CPU_BASE,
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

enum MaskOverlayMode
{
    MASK_OVERLAY_DISABLE = 0,
    MASK_OVERLAY_ENABLE,
};

//for VR entry exit points
enum ProxyGeometry
{
    PG_CUBE,
    PG_BRICKS,
};

#define BUFFER_BINDING_VISIBLE_LABEL_BUCKET 1
#define BUFFER_BINDING_VISIBLE_LABEL_ARRAY 2
#define BUFFER_BINDING_MASK_OVERLAY_COLOR_BUCKET 3

static const RGBUnit kColorTransverse = RGBUnit(237 , 25, 35);//ºá¶ÏÃæ ºìÉ«
static const RGBUnit kColorCoronal = RGBUnit(255 , 128 , 0);//¹Ú×´Ãæ ½Û»ÆÉ«
static const RGBUnit kColorSagittal = RGBUnit(1 , 255, 64);//Ê¸×´Ãæ ÂÌÉ«

MED_IMG_END_NAMESPACE
#endif