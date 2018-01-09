#include "mi_rc_step_mask_sampler.h"
#include "mi_shader_collection.h"

#include "glresource/mi_gl_buffer.h"

#include "mi_ray_caster.h"
#include "mi_ray_caster_inner_resource.h"

MED_IMG_BEGIN_NAMESPACE

GLShaderInfo RCStepMaskNoneSampler::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_MASK_NONE_SAMPLER_FRAG,
                        "RCStepMaskNoneSampler");
}

GLShaderInfo RCStepMaskNearstSampler::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_MASK_NEARST_SAMPLER_FRAG,
                        "RCStepMaskNearstSampler");
}

void RCStepMaskNearstSampler::set_gpu_parameter() {
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<RayCasterInnerResource> inner_buffer =
        ray_caster->get_inner_resource();

    GLBufferPtr buffer_visible_label =
        inner_buffer->get_buffer(RayCasterInnerResource::VISIBLE_LABEL_BUCKET);
    buffer_visible_label->bind_buffer_base(GL_SHADER_STORAGE_BUFFER,
                                           BUFFER_BINDING_VISIBLE_LABEL_BUCKET);
}

GLShaderInfo RCStepMaskLinearSampler::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_MASK_LINEAR_SAMPLER_FRAG,
                        "RCStepMaskNearstSampler");
}

void RCStepMaskLinearSampler::set_gpu_parameter() {
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<RayCasterInnerResource> inner_buffer =
        ray_caster->get_inner_resource();

    GLBufferPtr buffer_visible_label =
        inner_buffer->get_buffer(RayCasterInnerResource::VISIBLE_LABEL_BUCKET);
    buffer_visible_label->bind_buffer_base(GL_SHADER_STORAGE_BUFFER,
                                           BUFFER_BINDING_VISIBLE_LABEL_BUCKET);

    GLBufferPtr buffer_visible_label_array =
        inner_buffer->get_buffer(RayCasterInnerResource::VISIBLE_LABEL_ARRAY);
    buffer_visible_label_array->bind_buffer_base(GL_SHADER_STORAGE_BUFFER,
        BUFFER_BINDING_VISIBLE_LABEL_ARRAY);

    glUniform1i(_loc_visible_label_count,
        static_cast<int>(inner_buffer->get_visible_labels().size()));
}

void RCStepMaskLinearSampler::get_uniform_location() {
    GLProgramPtr program = _gl_program.lock();
    _loc_visible_label_count =
        program->get_uniform_location("visible_label_count");

    if (-1 == _loc_visible_label_count) {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}

MED_IMG_END_NAMESPACE
