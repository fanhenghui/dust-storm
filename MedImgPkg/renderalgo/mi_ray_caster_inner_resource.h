#ifndef MEDIMGRENDERALGO_RAY_CASTER_INNER_RESOUECE_H
#define MEDIMGRENDERALGO_RAY_CASTER_INNER_RESOUECE_H

#include "arithmetic/mi_color_unit.h"
#include "arithmetic/mi_vector2f.h"
#include "glresource/mi_gl_resource_define.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_gpu_resource_pair.h"

MED_IMG_BEGIN_NAMESPACE

class RayCasterInnerResource {
public:
    enum BufferType {
        WINDOW_LEVEL_BUCKET = 0,
        VISIBLE_LABEL_BUCKET,
        VISIBLE_LABEL_ARRAY,
        MASK_OVERLAY_COLOR_BUCKET,
        MATERIAL_BUCKET,
        TYPE_END,
    };

public:
    RayCasterInnerResource(GPUPlatform gpu_platform);

    ~RayCasterInnerResource();

    //----------------------------------------------//
    //GL interface
    GLBufferPtr get_buffer(BufferType type);

    //----------------------------------------------//
    //CUDA interface
    CudaGlobalMemoryPtr get_shared_map_memory();

    //----------------------------------------------//
    //input interface
    void set_mask_label_level(LabelLevel level);
    void set_window_level(float ww, float wl, unsigned char label);
    void set_visible_labels(std::vector<unsigned char> labels);
    void set_mask_overlay_color(std::map<unsigned char, RGBAUnit> colors);
    void set_mask_overlay_color(const RGBAUnit& color, unsigned char label);
    void set_material(const Material& matrial, unsigned char label);
    void set_color_opacity_texture_array(GPUTexture1DArrayPairPtr tex_array);//CUDA used only

    //----------------------------------------------//
    //get interface
    const std::vector<unsigned char>& get_visible_labels() const;
    const std::map<unsigned char, RGBAUnit>& get_mask_overlay_color() const;

private:
    bool check_dirty(BufferType type);
    void set_dirty(BufferType type);
    void remove_dirty(BufferType type);

private:
    GPUPlatform _gpu_platform;
    LabelLevel _label_level;

    struct GLResource;
    std::unique_ptr<GLResource> _inner_gl_resource;
    std::unique_ptr<char[]> _shared_buffer_array;

    struct CudaResource;
    std::unique_ptr<CudaResource> _inner_cuda_resource;    

    std::vector<unsigned char> _labels;
    std::map<unsigned char, Vector2f> _window_levels;
    std::map<unsigned char, RGBAUnit> _mask_overlay_colors;
    std::map<unsigned char, Material> _material;
    GPUTexture1DArrayPairPtr _color_opacity_tex_array;
};

MED_IMG_END_NAMESPACE

#endif