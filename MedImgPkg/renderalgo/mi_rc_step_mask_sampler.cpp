#include "mi_rc_step_mask_sampler.h"
#include "mi_shader_collection.h"

#include "glresource/mi_gl_buffer.h"

#include "mi_ray_caster.h"
#include "mi_ray_caster_inner_buffer.h"

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
  std::shared_ptr<RayCasterInnerBuffer> inner_buffer =
      ray_caster->get_inner_buffer();

  GLBufferPtr buffer_visible_label =
      inner_buffer->get_buffer(RayCasterInnerBuffer::VISIBLE_LABEL_BUCKET);
  buffer_visible_label->bind_buffer_base(GL_SHADER_STORAGE_BUFFER,
                                         BUFFER_BINDING_VISIBLE_LABEL_BUCKET);
}

MED_IMG_END_NAMESPACE
