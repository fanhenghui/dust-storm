#version 430

float access_volume(sampler3D sampler , vec3 vPos)
{
    return texture(sampler , vPos).r;
}