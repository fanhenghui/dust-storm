#ifndef MEDIMGRENDERALGO_RAY_CASTER_DEFINE_H
#define MEDIMGRENDERALGO_RAY_CASTER_DEFINE_H

#include "renderalgo/mi_render_algo_export.h"
#include "arithmetic/mi_color_unit.h"
#include "arithmetic/mi_cuda_graphic.h"
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

//------------------------------------------------------//
//For CUDA Ray Casting
//------------------------------------------------------//
struct CudaRayCastInfos {
    //label level to restrict max rc-mask label number .
    //note: 1 for non-mask; max 256
    int label_level;

    //ray-casting mode
    int mask_mode;
    int composite_mode;
    int interpolation_mode;
    int shading_mode;
    int color_inverse_mode;
    int mask_overlay_mode;

    //sample step
    float sample_step;

    //illumination parameters
    mat4 mat_normal;//transpose(inverse(mat_m2v))
    float3 light_position;//point light based
    float3 ambient_color;//ambient RGB normalized
    float ambient_intensity;//ambient intensity

    //global window level(MIP MinIP Average)
    float global_ww;
    float global_wl;

    //transfer function parameters
    float color_opacity_texture_shift;

    //pseudo color texture
    cudaTextureObject_t pseudo_color_texture;
    float pseudo_color_texture_shift;

    //---------------------------------------------------------//
    //shared mapped global memory contains follows:
    // 1. visible label (int) : label_level * sizeof(int), label_level could be 1(none-mask) 8 16 32 64 ... 128
    // 2. ww wl array (flaot) : label_level * sizeof(float) * 2
    // 3. color/opacity texture array (tex1D): label_level * sizeof(unsigned long long)
    // 4. materal parameter : label_level * sizeof(float) * 9
    // sum: label_level * [4*1 + 4*2 + 8*1 + 4*9] = label_level * 56, max : 14KB < shared limits(40KB)
    //---------------------------------------------------------//
    void* d_shared_mapped_memory;
    
    //test code
    //0 non-test
    //1 show entry points
    //2 show exit points
    int test_code;

    CudaRayCastInfos() {
        label_level = 1;
        
        mask_mode = 0;
        composite_mode = 0;
        interpolation_mode = 0;
        shading_mode = 0;
        color_inverse_mode = 0;
        mask_overlay_mode = 0;

        sample_step = 0.5f;

        mat_normal = matrix4_to_mat4(medical_imaging::Matrix4::S_IDENTITY_MATRIX);
        light_position = make_float3(0.0f);
        ambient_color = make_float3(1.0f);
        ambient_intensity = 0.3f;

        global_ww = 0.0f;
        global_wl = 0.0f;

        color_opacity_texture_shift = 0.5f/512.0f;

        pseudo_color_texture = 0; 
        pseudo_color_texture_shift = 0.5f/512.0f;
        d_shared_mapped_memory = nullptr;

        test_code = 0;
    }
};

struct CudaVolumeInfos {
    cudaTextureObject_t volume_tex;
    cudaTextureObject_t mask_tex;
    uint3 dim;
    float3 dim_r;
    float3 sample_shift;

    CudaVolumeInfos() {
        volume_tex = 0;
        mask_tex = 0;
        dim = make_uint3(0);
        dim_r = make_float3(0.0f);
        sample_shift = make_float3(0.0f);
    }
};

#endif