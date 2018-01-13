#include "mi_rc_step_composite.h"
#include "mi_shader_collection.h"

#include "glresource/mi_gl_buffer.h"
#include "glresource/mi_gl_program.h"
#include "glresource/mi_gl_texture_1d_array.h"

#include "io/mi_image_data.h"

#include "mi_ray_caster.h"
#include "mi_ray_caster_inner_resource.h"

MED_IMG_BEGIN_NAMESPACE

GLShaderInfo RCStepCompositeAverage::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_COMPOSITE_AVERAGE_FRAG,
                        "RCStepCompositeAverage");
}

GLShaderInfo RCStepCompositeMIP::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_COMPOSITE_MIP_FRAG,
                        "RCStepCompositeMIP");
}

GLShaderInfo RCStepCompositeMinIP::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_COMPOSITE_MINIP_FRAG,
                        "RCStepCompositeMinIP");
}

void RCStepCompositeMinIP::set_gpu_parameter() {}

void RCStepCompositeMinIP::get_uniform_location() {
    GLProgramPtr program = _gl_program.lock();
    _loc_custom_min_threshold =
        program->get_uniform_location("custom_min_threshold");

    if (-1 == _loc_custom_min_threshold) {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}

GLShaderInfo RCStepCompositeDVR::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_COMPOSITE_DVR_FRAG,
                        "RCStepCompositeDVR");
}

void RCStepCompositeDVR::set_gpu_parameter() {
    CHECK_GL_ERROR;

    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<RayCasterInnerResource> inner_buffer =
        ray_caster->get_inner_resource();

    GLBufferPtr buffer_wl =
        inner_buffer->get_buffer(RayCasterInnerResource::WINDOW_LEVEL_BUCKET);
    buffer_wl->bind_buffer_base(GL_SHADER_STORAGE_BUFFER,
                                BUFFER_BINDING_WINDOW_LEVEL_BUCKET);

    GLTexture1DArrayPtr color_opacity_tex_array =
        ray_caster->get_color_opacity_texture_array()->get_gl_resource();
    const int act_tex = _act_tex_counter->tick();
    glActiveTexture(GL_TEXTURE0 + act_tex);
    color_opacity_tex_array->bind();
    GLTextureUtils::set_1d_array_wrap_s(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_1D_ARRAY, GL_LINEAR);
    glUniform1i(_loc_color_opacity_array, act_tex);

    glUniform1f(_loc_color_opacity_texture_shift, 0.5f / S_TRANSFER_FUNC_WIDTH);

    glUniform1f(_loc_sample_step, ray_caster->get_sample_step());

    std::shared_ptr<ImageData> volume_img = ray_caster->get_volume_data();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_img);
    const double min_spacing = (std::min)((std::min)(volume_img->_spacing[0], 
        volume_img->_spacing[1]), volume_img->_spacing[2]);
    glUniform1f(_loc_opacity_correction, static_cast<float>(min_spacing));

    CHECK_GL_ERROR;
}

void RCStepCompositeDVR::get_uniform_location() {
    GLProgramPtr program = _gl_program.lock();
    _loc_color_opacity_array =
        program->get_uniform_location("color_opacity_array");
    _loc_color_opacity_texture_shift =
        program->get_uniform_location("color_opacity_texture_shift");
    _loc_opacity_correction = program->get_uniform_location("opacity_correction");
    _loc_sample_step = program->get_uniform_location("sample_step");

    if (-1 == _loc_color_opacity_array ||
            -1 == _loc_color_opacity_texture_shift || -1 == _loc_opacity_correction ||
            -1 == _loc_sample_step) {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}

MED_IMG_END_NAMESPACE