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
        WindowLevelBucket = 0,
        VisibleLabelBucket,
        TypeEnd,
    };

    LabelLevel m_eLabelLevel;

    std::vector<unsigned char> m_vecVisibleLabel;

    std::map<unsigned char, Vector2f> m_mapWindowLevel;

public:
    RayCasterInnerBuffer();

    ~RayCasterInnerBuffer();

    void release_buffer();

    GLBufferPtr get_buffer(BufferType eType);

    void set_mask_label_level(LabelLevel eLevel);

    void set_window_level(float fWW , float fWL , unsigned char ucLabel);

    void set_visible_labels(std::vector<unsigned char> vecLabels);

private:
    struct GLResource;
    std::unique_ptr<GLResource> m_pRes;

    std::unique_ptr<char[]> m_pSharedBufferArray;
};

MED_IMAGING_END_NAMESPACE

#endif