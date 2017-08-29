#include "mi_rc_step_shading.h"

#include "arithmetic/mi_camera_base.h"
#include "glresource/mi_gl_buffer.h"
#include "io/mi_image_data.h"

#include "mi_camera_calculator.h"
#include "mi_ray_caster.h"
#include "mi_ray_caster_inner_buffer.h"
#include "mi_shader_collection.h"

MED_IMG_BEGIN_NAMESPACE

GLShaderInfo RCStepShadingNone::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_SHADING_NONE_FRAG,
                        "RCStepShadingNone");
}

GLShaderInfo RCStepShadingPhong::get_shader_info() {
    return GLShaderInfo(GL_FRAGMENT_SHADER, S_RC_SHADING_PHONG_FRAG,
                        "RCStepShadingPhong");
}

void RCStepShadingPhong::get_uniform_location() {
    GLProgramPtr program = _gl_program.lock();
    _loc_mat_normal = program->get_uniform_location("mat_normal");
    _loc_spacing = program->get_uniform_location("spacing");
    _loc_light_position = program->get_uniform_location("light_position");
    _loc_ambient_color = program->get_uniform_location("ambient_color");

    if (-1 == _loc_mat_normal || -1 == _loc_spacing ||
            -1 == _loc_light_position || -1 == _loc_ambient_color) {
        RENDERALGO_THROW_EXCEPTION("Get uniform location failed!");
    }
}

void RCStepShadingPhong::set_gpu_parameter() {
    CHECK_GL_ERROR;

    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<RayCasterInnerBuffer> inner_buffer =
        ray_caster->get_inner_buffer();
    std::shared_ptr<CameraCalculator> camera_cal =
        ray_caster->get_camera_calculator();
    std::shared_ptr<CameraBase> camera = ray_caster->get_camera();

    std::shared_ptr<ImageData> volume_img = ray_caster->get_volume_data();
    const float spacing[3] = {static_cast<float>(volume_img->_spacing[0]),
                              static_cast<float>(volume_img->_spacing[1]),
                              static_cast<float>(volume_img->_spacing[2])
                             };
    glUniform3f(_loc_spacing, spacing[0], spacing[1], spacing[2]);

    const Matrix4 mat_view = camera->get_view_matrix();
    const Matrix4 mat_v2w = camera_cal->get_volume_to_world_matrix();
    const Matrix4 mat_v2view = mat_view * mat_v2w;
    const Matrix4 mat_normal = mat_v2view.get_inverse().get_transpose();
    float mat[16];
    mat_normal.to_float16(mat);
    glUniformMatrix4fv(_loc_mat_normal, 1, GL_FALSE, mat);

    Point3 eye = camera->get_eye();
    Point3 lookat = camera->get_look_at();
    Vector3 view = camera->get_view_direction();
    double max_dim =
        (std::max)((std::max)(volume_img->_dim[0] * volume_img->_spacing[0],
                              volume_img->_dim[1] * volume_img->_spacing[1]),
                   volume_img->_dim[2] * volume_img->_spacing[2]);
    const float magic_num = 1.5f;
    Point3 light_pos = lookat - view * max_dim * magic_num;
    light_pos = camera_cal->get_world_to_volume_matrix().transform(light_pos);
    glUniform3f(_loc_light_position, static_cast<float>(light_pos.x),
                static_cast<float>(light_pos.y), static_cast<float>(light_pos.z));

    float ambient_color[4] = {0, 0, 0, 0};
    ray_caster->get_ambient_color(ambient_color);
    glUniform4f(_loc_ambient_color, ambient_color[0], ambient_color[1],
                ambient_color[2], ambient_color[3]);

    GLBufferPtr buffer =
        inner_buffer->get_buffer(RayCasterInnerBuffer::MATERIAL_BUCKET);
    buffer->bind_buffer_base(GL_SHADER_STORAGE_BUFFER,
                             BUFFER_BINDING_MATERIAL_BUCKET);

    CHECK_GL_ERROR;
}

MED_IMG_END_NAMESPACE
