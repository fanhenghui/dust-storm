#include "arithmetic/mi_cuda_graphic.h"

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
    //opacity correction for adaptive sample step
    float opacity_correction;

    //illumination parameters
    mat4 mat_normal;//transpose(inverse(mat_m2v))
    float3 light_position;//point light based
    float3 ambient_color;//ambient RGB normalized
    float ambient_intensity;//ambient intensity

    //global window level(MIP MinIP Average)
    float global_ww;
    float global_wl;

    //MinIP threshold
    float minip_threshold;

    //transfer function parameters
    float color_opacity_texture_shift;

    //pseudo color texture
    cudaTextureObject_t pseudo_color_texture;
    float pseudo_color_texture_shift;

    //mask overlay
    float mask_overlay_opacity;

    //jittering
    int jittering;
    cudaTextureObject_t random_texture;

    //ray align to view plane
    int ray_align_to_view_plane;
    float3 eye_position;

    //---------------------------------------------------------//
    //shared mapped global memory contains follows:
    // 1. visible label (int) : label_level * sizeof(int), label_level could be 1(none-mask) 8 16 32 64 ... 128
    // 2. ww wl array (flaot) : label_level * sizeof(float) * 2
    // 3. color/opacity texture array (tex1D): label_level * sizeof(unsigned long long)
    // 4. materal parameter : label_level * sizeof(float) * 9
    // 5. mask overlay color: label_level * sizeof(float) * 4
    // sum: label_level * [4*1 + 4*2 + 8*1 + 4*9 + 4*4] = label_level * 72, max : 18KB < shared limits(40KB)
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
        opacity_correction = 0.5f;

        mat_normal = matrix4_to_mat4(medical_imaging::Matrix4::S_IDENTITY_MATRIX);
        light_position = make_float3(0.0f);
        ambient_color = make_float3(1.0f);
        ambient_intensity = 0.3f;

        global_ww = 0.0f;
        global_wl = 0.0f;
        
        minip_threshold = 0.0f;

        color_opacity_texture_shift = 0.5f/512.0f;

        pseudo_color_texture = 0; 
        pseudo_color_texture_shift = 0.5f/512.0f;

        mask_overlay_opacity = 0.5f;

        jittering = 0;
        random_texture = 0;

        ray_align_to_view_plane = 0;
        eye_position = make_float3(0.0f);

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