#version 430

bool check_opacity(in out float opacity);

void composite(vec3 samplePosVolume, vec3 ray_dir, in out vec4 vIntegralColor,
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);

vec4 raycast(vec3 vRayStart, vec3 ray_dir, float fStartStep, float fEndStep, vec4 vIntegralColor,
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift, int iRayCastStepCode)
{

}
