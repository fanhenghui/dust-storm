#version 430

bool access_mask(sampler3D sampler , vec3 pos , out int out_label)
{
    out_label = 0;
    return true;
}
