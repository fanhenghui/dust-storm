#version 430

float AccessVolume(sampler3D sampler , vec3 vPos)
{
    return texture(sampler , vPos).r;
}