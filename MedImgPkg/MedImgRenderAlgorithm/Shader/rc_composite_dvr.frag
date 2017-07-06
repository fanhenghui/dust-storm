#version 430
#extension GL_EXT_texture_array : enable

#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 5

uniform sampler1DArray color_table_array; 
uniform float color_table_texture_shift;
uniform float opacity_compensation; 
uniform float sample_rate;

//Window level buffer
layout (std430 , binding = BUFFER_BINDING_WINDOW_LEVEL_BUCKET) buffer WindowLevelBucket
{
    vec2 windowing[];
};

bool access_mask(vec3 pos, sampler3D sampler , out int out_label);
float access_volume(sampler3D sampler , vec3 pos);

vec4 shade(vec3 sample_pos, vec4 output_color, vec3 ray_dir , sampler3D sampler , vec3 pos_in_volume , vec3 sample_shift , int label);

void composite(vec3 sample_pos,vec3 ray_dir, in out vec4 integral_color, 
sampler3D volume_sampler , sampler3D mask_sampler , vec3 sub_data_dim , vec3 sub_data_offset , vec3 sample_shift)
{
    int label = 0;
    vec4 current_color = vec4(0.0,0.0,0.0,0.0);
    float wl_min_gray;
    float gray;

    vec3 actual_sample_pos = (sample_pos + sub_data_offset )/sub_data_dim;//Actual SamplePos in sampler
    if(access_mask(mask_sampler , actual_sample_pos , label))
    {
        wl_min_gray = windowing[label].y - 0.5 * windowing[label].x;

        gray = access_volume(volume_sampler, actual_sample_pos);
        gray = (gray - wl_min_gray) / windowing[label].x;

        current_color = texture1DArray(color_table_array, vec2(gray + color_table_texture_shift , label) );
        if(current_color.a >0.0)
        {
            vec4 shading_color = shade(actual_sample_pos, current_color, ray_dir , volume_sampler , sample_pos , sample_shift , label);
            shading_color.a = 1 - pow(1 - shading_color.a, sample_rate/opacity_compensation);
            integral_color.rgb += shading_color.rgb * (1.0 - integral_color.a) * shading_color.a;
            integral_color.a += shading_color.a * (1.0 - integral_color.a);
         }
    }
}
