#version 430

vec4 Shade(vec3 sampleCoord, vec4 sampleColor, vec3 rayDir , sampler3D vDataVolume , vec3 samplePosVolume , vec3 vSampleShift , int idx)
{
    return sampleColor;
}