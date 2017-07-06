#version 430

float access_volume(sampler3D sampler , vec3 pos)
{
    return texture(sampler , pos).r;
}