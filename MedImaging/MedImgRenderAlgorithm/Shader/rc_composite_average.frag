#version 430

bool access_mask(sampler3D sampler , vec3 pos , out int out_label);
float access_volume(sampler3D sampler , vec3 pos);

void composite(vec3 sample_pos, vec3 ray_dir, in out vec4 integral_color,
    sampler3D volume_sampler  , sampler3D mask_sampler , vec3 sub_data_dim , vec3 sub_data_offset,  vec3 sample_shift)
{
    int label = 0;
    vec3 actual_sample_pos = (sample_pos + sub_data_offset)/sub_data_dim;
    if(access_mask(mask_sampler , actual_sample_pos , label))
    {
        float gray = access_volume(volume_sampler, actual_sample_pos);
        integral_color = vec4(gray,gray,gray,1.0);
    }
}
