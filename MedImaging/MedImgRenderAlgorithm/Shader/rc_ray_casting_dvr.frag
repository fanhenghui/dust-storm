#version 430

bool check_opacity(in out float opacity);

void composite(vec3 sample_pos_volume, vec3 ray_dir, in out vec4 integral_color,
    sampler3D volume_sampler  , sampler3D mask_sampler , vec3 sub_data_dim , vec3 sub_data_offset,  vec3 sample_shift);

vec4 raycast(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
    sampler3D volume_sampler,  sampler3D mask_sampler,   vec3 sub_data_dim , vec3 sub_data_offset , vec3 sample_shift, int ray_cast_step_code)
{

}
