#version 430

uniform sampler1D pseudo_color;
uniform float pseudo_color_slope;
uniform float pseudo_color_intercept;

uniform vec2 global_wl;

float global_max_gray = -65535.0;
float global_min_gray = 65535.0;
vec3 global_target_pos = vec3(0,0,0);

void composite(vec3 sample_pos_volume, vec3 ray_dir, in out vec4 integral_color,
    sampler3D volume_sampler, sampler3D mask_sampler, vec3 sub_data_dim, vec3 sub_data_offset, vec3 sample_shift);

void color_inverse(in out float gray);

vec4 ray_cast(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
    sampler3D volume_sampler, sampler3D mask_sampler, vec3 sub_data_dim, vec3 sub_data_offset, vec3 sample_shift, 
    int ray_cast_step_code, out vec3 ray_end)
{
    if(0!= (ray_cast_step_code & 0x0001))//First sub data
    {
        global_max_gray = -65535.0;
        global_min_gray = 65535.0;
    }

    if(0!= (ray_cast_step_code & 0x0002))//Middle sub data
    {
        global_max_gray = integral_color.r;
        global_min_gray = integral_color.r;
    }

    vec3 sample_pos;
    vec4 current_integral_color =  integral_color;

    for (float i = start_step ; i < end_step ; ++i)
    {
        sample_pos = ray_start + ray_dir * i;
        composite(sample_pos , ray_dir, current_integral_color , volume_sampler , mask_sampler , sub_data_dim , sub_data_offset ,sample_shift );
    }

    //Last sub data transfer gray to color
    if(0!= (ray_cast_step_code & 0x0004))
    {
        //check ray cast through air
        if (global_target_pos == vec3(0, 0, 0)) {
            discard;
        }

        float ww = global_wl.x;
        float wl = global_wl.y;
        float result_gray = current_integral_color.r;
        float wl_min_gray = wl - 0.5 * ww;
        result_gray= (result_gray - wl_min_gray) / ww;
        result_gray = clamp(result_gray, 0.0, 1.0);
        if(result_gray < 0.000001)
        {
            discard;
        }

        color_inverse(result_gray);
        current_integral_color = vec4(texture(pseudo_color, (result_gray*pseudo_color_slope + pseudo_color_intercept)).rgb, 1.0);
        ray_end = global_target_pos;
    }

    return current_integral_color;
}
