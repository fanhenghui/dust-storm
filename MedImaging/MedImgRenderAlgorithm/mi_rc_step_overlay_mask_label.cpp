#include "mi_rc_step_overlay_mask_label.h"
#include "mi_shader_collection.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"
#include "mi_ray_caster_inner_buffer.h"

MED_IMAGING_BEGIN_NAMESPACE



GLShaderInfo RCStepOverlayMaskLabelEnable::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_OVERLAY_MASK_LABEL_ENABLE_FRAG , "RCStepOverlayMaskLabelEnableFrag");
}

void RCStepOverlayMaskLabelEnable::set_gpu_parameter()
{
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<RayCasterInnerBuffer> inner_buffer = ray_caster->get_inner_buffer();

    GLBufferPtr buffer_mask_overlay = inner_buffer->get_buffer(RayCasterInnerBuffer::MASK_OVERLAY_COLOR_BUCKET);
    buffer_mask_overlay->bind_buffer_base(GL_SHADER_STORAGE_BUFFER , BUFFER_BINDING_MASK_OVERLAY_COLOR_BUCKET);

    GLBufferPtr buffer_visible_label = inner_buffer->get_buffer(RayCasterInnerBuffer::VISIBLE_LABEL_ARRAY);
    buffer_visible_label->bind_buffer_base(GL_SHADER_STORAGE_BUFFER , BUFFER_BINDING_VISIBLE_LABEL_ARRAY);

    glUniform1i(_loc_visible_label_count , static_cast<int>(inner_buffer->get_visible_labels().size()));
}

void RCStepOverlayMaskLabelEnable::get_uniform_location()
{
    GLProgramPtr program = _program.lock();
    _loc_visible_label_count = program->get_uniform_location("visible_label_count");

    if (-1 == _loc_visible_label_count)
    {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}


GLShaderInfo RCStepOverlayMaskLabelDisable::get_shader_info()
{
    return GLShaderInfo(GL_FRAGMENT_SHADER , S_RC_OVERLAY_MASK_LABEL_DISABLE_FRAG , "RCStepOverlayMaskLabelDisableFrag");
}

MED_IMAGING_END_NAMESPACE