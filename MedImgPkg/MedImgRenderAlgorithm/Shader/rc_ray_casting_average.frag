#version 430

#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 5

uniform sampler1D pseudo_color;
uniform float pseudo_color_slope;
uniform float pseudo_color_intercept;

uniform vec2 global_wl;

void composite(vec3 sample_pos_volume, vec3 ray_dir, in out vec4 integral_color,
    sampler3D volume_sampler  , sampler3D mask_sampler , vec3 sub_data_dim , vec3 sub_data_offset,  vec3 sample_shift);

void color_inverse(in out float gray);

vec4 ray_cast(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
    sampler3D volume_sampler,  sampler3D mask_sampler,   vec3 sub_data_dim , vec3 sub_data_offset , vec3 sample_shift , int ray_cast_step_code)
{
    float sum_gray = 0.0;
    float sum_num = 0.0;

    if(0 != (ray_cast_step_code & 0x0001))//First sub data
    {
        sum_gray = 0.0;
        sum_num = 0.0;
    }

    if(0 != (ray_cast_step_code & 0x0002))//Middle sub data
    {
        sum_gray = integral_color.r;
        sum_num = integral_color.g;
    }

    vec3 sample_pos;
    vec4 current_integral_color =  integral_color;

    for (float i = start_step ; i <= end_step ; ++i)
    {
        sample_pos = ray_start + ray_dir * i;
        composite(sample_pos , ray_dir, current_integral_color , volume_sampler , mask_sampler , sub_data_dim ,sub_data_offset ,sample_shift );
        sum_gray += current_integral_color.r*100.0;
        ++sum_num;
    }

    current_integral_color = vec4(sum_gray , sum_num , 0 ,0);

    //Last sub data transfer gray to color
    if(0 != (ray_cast_step_code & 0x0004))
    {
        float ww = global_wl.x;
        float wl = global_wl.y;
        float result_gray = sum_gray/sum_num/100.0;
        float wl_min_gray = wl - 0.5 * ww;
        result_gray= (result_gray - wl_min_gray) / ww;
        result_gray = clamp(result_gray, 0.0, 1.0);
        if(result_gray < 0.000001)
        {
            discard;
        }

        color_inverse(result_gray);
        current_integral_color = vec4(texture(pseudo_color, (result_gray*pseudo_color_slope + pseudo_color_intercept)).rgb, 1.0);
    }

    return current_integral_color;

}