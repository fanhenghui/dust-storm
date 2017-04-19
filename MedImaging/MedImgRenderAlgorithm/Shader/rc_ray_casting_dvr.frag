#version 430

bool CheckOpacity(in out float opacity);

void Composite(vec3 samplePosVolume, vec3 vRayDir, in out vec4 vIntegralColor,
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);

vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift, int iRayCastStepCode)
{

}
