#version 430

float access_volume(sampler3D sampler , vec3 vPos)
{
    vec3 vDim = textureSize(sampler, 0);
    return texturefetch(sampler , ivec3(vPos*vDim) , 0 ).r;
}