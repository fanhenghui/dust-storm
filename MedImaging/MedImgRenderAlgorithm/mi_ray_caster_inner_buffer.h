#ifndef MED_IMAGING_RAY_CASTER_INNER_BUFFER_H_
#define MED_IMAGING_RAY_CASTER_INNER_BUFFER_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgArithmetic/mi_vector2f.h"
#include "MedImgGLResource/mi_gl_resource_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RayCasterInnerBuffer
{
public:
    enum BufferType
    {
        WINDOW_LEVEL_BUCKET = 0,
        VISIBLE_LABEL_BUCKET,
        TYPE_END,
    };

    LabelLevel _label_level;

    std::vector<unsigned char> _labels;

    std::map<unsigned char, Vector2f> _window_levels;

public:
    RayCasterInnerBuffer();

    ~RayCasterInnerBuffer();

    void release_buffer();

    GLBufferPtr get_buffer(BufferType type);

    void set_mask_label_level(LabelLevel level);

    void set_window_level(float ww , float wl , unsigned char label);

    void set_visible_labels(std::vector<unsigned char> labels);

private:
    struct GLResource;
    std::unique_ptr<GLResource> _inner_resource;

    std::unique_ptr<char[]> _shared_buffer_array;
};

MED_IMAGING_END_NAMESPACE

#endif