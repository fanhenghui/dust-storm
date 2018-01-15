#version 430

float global_max_gray;
vec3 global_target_pos;

bool access_mask(sampler3D sampler , vec3 pos , out int out_label);
float access_volume(sampler3D sampler , vec3 pos);

void composite(vec3 sample_pos, vec3 ray_dir, in out vec4 integral_color,
    sampler3D volume_sampler  , sampler3D mask_sampler , vec3 sub_data_dim , vec3 sub_data_offset,  vec3 sample_shift)
{
    int label = 0;
    vec3 actual_sample_pos = (sample_pos + sub_data_offset + vec3(0.5,0.5,0.5))/sub_data_dim;
    if(access_mask(mask_sampler , actual_sample_pos , label))
    {
        float gray = access_volume(volume_sampler, actual_sample_pos);
        if(global_max_gray < gray)
        {
            global_max_gray = gray;
            integral_color = vec4(gray,gray,gray,1.0);
            global_target_pos = sample_pos;
        }
    }
}
