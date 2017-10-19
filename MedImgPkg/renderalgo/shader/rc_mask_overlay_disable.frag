#version 430

vec4 mask_overlay(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
        sampler3D volume_sampler, sampler3D mask_sampler, vec3 sub_data_dim, vec3 sub_data_offset, int ray_cast_step_code)
{
    return integral_color;
}
