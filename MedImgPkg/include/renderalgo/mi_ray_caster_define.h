#ifndef MEDIMGRENDERALGO_RAY_CASTER_DEFINE_H
#define MEDIMGRENDERALGO_RAY_CASTER_DEFINE_H

#include "renderalgo/mi_render_algo_export.h"
#include "arithmetic/mi_color_unit.h"
#include "io/mi_io_define.h"

MED_IMG_BEGIN_NAMESPACE

//Branch Master is all GPU
enum RayCastingStrategy {
    CPU_BASE,
    GPU_BASE,
};

enum GPUPlatform {
    GL_BASE,
    CUDA_BASE,
};

enum LabelLevel {
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

enum InterpolationMode {
    NEARST = 0,
    LINEAR = 1,
    CUBIC = 2,
};

enum CompositeMode {
    COMPOSITE_DVR = 0,
    COMPOSITE_MIP = 1,
    COMPOSITE_MINIP = 2,
    COMPOSITE_AVERAGE = 3,
};

enum MaskMode {
    MASK_NONE = 0,
    MASK_MULTI_LABEL = 1,
    MASK_MULTI_LINEAR_LABEL = 2,
};

enum ShadingMode {
    SHADING_NONE = 0,
    SHADING_PHONG = 1,
};

enum ColorInverseMode {
    COLOR_INVERSE_DISABLE = 0,
    COLOR_INVERSE_ENABLE = 1,
};

enum MaskOverlayMode {
    MASK_OVERLAY_DISABLE = 0,
    MASK_OVERLAY_ENABLE = 1,
};

// for VR entry exit points
enum ProxyGeometry {
    PG_CUBE = 0,
    PG_BRICKS = 1,
};

#define BUFFER_BINDING_VISIBLE_LABEL_BUCKET 1
#define BUFFER_BINDING_VISIBLE_LABEL_ARRAY 2
#define BUFFER_BINDING_MASK_OVERLAY_COLOR_BUCKET 3
#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 4
#define BUFFER_BINDING_MATERIAL_BUCKET 5

static const RGBUnit kColorTransverse = RGBUnit(237, 25, 35);
static const RGBUnit kColorCoronal = RGBUnit(255, 128, 0);
static const RGBUnit kColorSagittal = RGBUnit(1, 255, 64);

static const int S_TRANSFER_FUNC_WIDTH = 512;

struct Material {
    float diffuse[4];
    float specular[4];
    float specular_shiness;
    float reserve0;
    float reserve1;
    float reserve2;

    bool operator==(const Material& m) const {
        return (fabs(diffuse[0] - m.diffuse[0]) < FLOAT_EPSILON &&
                fabs(diffuse[1] - m.diffuse[1]) < FLOAT_EPSILON &&
                fabs(diffuse[2] - m.diffuse[2]) < FLOAT_EPSILON &&
                fabs(diffuse[3] - m.diffuse[3]) < FLOAT_EPSILON &&
                fabs(specular[0] - m.specular[0]) < FLOAT_EPSILON &&
                fabs(specular[1] - m.specular[1]) < FLOAT_EPSILON &&
                fabs(specular[2] - m.specular[2]) < FLOAT_EPSILON &&
                fabs(specular[3] - m.specular[3]) < FLOAT_EPSILON &&
                fabs(specular_shiness - m.specular_shiness) < FLOAT_EPSILON);
    }

    bool operator!=(const Material& m) const {
        return (fabs(diffuse[0] - m.diffuse[0]) > FLOAT_EPSILON ||
                fabs(diffuse[1] - m.diffuse[1]) > FLOAT_EPSILON ||
                fabs(diffuse[2] - m.diffuse[2]) > FLOAT_EPSILON ||
                fabs(diffuse[3] - m.diffuse[3]) > FLOAT_EPSILON ||
                fabs(specular[0] - m.specular[0]) > FLOAT_EPSILON ||
                fabs(specular[1] - m.specular[1]) > FLOAT_EPSILON ||
                fabs(specular[2] - m.specular[2]) > FLOAT_EPSILON ||
                fabs(specular[3] - m.specular[3]) > FLOAT_EPSILON ||
                fabs(specular_shiness - m.specular_shiness) > FLOAT_EPSILON);
    }
};

MED_IMG_END_NAMESPACE

#endif