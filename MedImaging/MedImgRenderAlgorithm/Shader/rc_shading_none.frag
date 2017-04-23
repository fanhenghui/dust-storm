#version 430

vec4 shade(vec3 sampleCoord, vec4 sampleColor, vec3 ray_dir , sampler3D vDataVolume , vec3 samplePosVolume , vec3 vSampleShift , int idx)
{
    return sampleColor;
}