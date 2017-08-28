#version 430

bool check_opacity(in out float opacity);

void composite(vec3 sample_pos_volume, vec3 ray_dir, in out vec4 integral_color,
    sampler3D volume_sampler  , sampler3D mask_sampler , vec3 sub_data_dim , vec3 sub_data_offset,  vec3 sample_shift);

vec4 ray_cast(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
    sampler3D volume_sampler,  sampler3D mask_sampler,   vec3 sub_data_dim , vec3 sub_data_offset , vec3 sample_shift, int ray_cast_step_code)
{
    if(0!= (ray_cast_step_code & 0x0001))//First sub data
    {
        integral_color.a = 0;
    }

    if(0!= (ray_cast_step_code & 0x0002))//Middle sub data
    {

    }

    vec3 sample_pos;
    vec4 current_integral_color =  integral_color;

    for (float i = start_step ; i <= end_step ; ++i)
    {
        sample_pos = ray_start + ray_dir * i;
        composite(sample_pos , ray_dir, current_integral_color , volume_sampler , mask_sampler , sub_data_dim , sub_data_offset ,sample_shift );
        if(check_opacity(current_integral_color.a))
        {
            break;
        }
    }

    //if(0!= (ray_cast_step_code & 0x0004))
    //{
    //    current_integral_color = clamp(current_integral_color, 0.0, 1.0);
    //}

    return current_integral_color;
}
