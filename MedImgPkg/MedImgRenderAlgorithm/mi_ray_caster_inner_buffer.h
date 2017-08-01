#ifndef MED_IMG_RAY_CASTER_INNER_BUFFER_H_
#define MED_IMG_RAY_CASTER_INNER_BUFFER_H_

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgArithmetic/mi_vector2f.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgArithmetic/mi_color_unit.h"

MED_IMG_BEGIN_NAMESPACE

class RayCasterInnerBuffer
{
public:
    enum BufferType
    {
        WINDOW_LEVEL_BUCKET = 0,
        VISIBLE_LABEL_BUCKET,
        VISIBLE_LABEL_ARRAY,
        MASK_OVERLAY_COLOR_BUCKET,
        TYPE_END,
    };

    LabelLevel _label_level;

    std::vector<unsigned char> _labels;

    std::map<unsigned char, Vector2f> _window_levels;

    std::map<unsigned char , RGBAUnit> _mask_overlay_colors;

public:
    RayCasterInnerBuffer();

    ~RayCasterInnerBuffer();

    GLBufferPtr get_buffer(BufferType type);

    void set_mask_label_level(LabelLevel level);

    void set_window_level(float ww , float wl , unsigned char label);

    void set_visible_labels(std::vector<unsigned char> labels);
    const std::vector<unsigned char>& get_visible_labels() const;

    void set_mask_overlay_color(std::map<unsigned char , RGBAUnit> colors);
    void set_mask_overlay_color(RGBAUnit color , unsigned char label);
    const std::map<unsigned char , RGBAUnit>& get_mask_overlay_color() const;

private:
    struct GLResource;
    std::unique_ptr<GLResource> _inner_resource;

    std::unique_ptr<char[]> _shared_buffer_array;


};

MED_IMG_END_NAMESPACE

#endif