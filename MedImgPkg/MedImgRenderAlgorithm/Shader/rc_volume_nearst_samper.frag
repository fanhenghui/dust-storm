#version 430

float access_volume(sampler3D sampler , vec3 pos)
{
    vec3 dim = textureSize(sampler, 0);
    return texturefetch(sampler , ivec3(pos*dim) , 0 ).r;
}