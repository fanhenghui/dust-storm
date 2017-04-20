#version 430

float fGlobalMinGray;

uniform float fCustomMinThreshold;

bool access_mask(sampler3D sampler , vec3 vPos , out int iOutLabel);
float access_volume(sampler3D sampler , vec3 vPos);

void composite(vec3 vSamplePos, vec3 vRayDir, in out vec4 vIntegralColor,
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift)
{
    int iLabel = 0;
    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset)/vSubDataDim;
    if(access_mask(sMask , vActualSamplePos , iLabel))
    {
        float fGray = access_volume(sVolume, vActualSamplePos);
        if (fGray > fCustomMinThreshold && fGray < fGlobalMinGray)
        { 
            fGlobalMinGray  = fGray;
            vIntegralColor = vec4(fGray,fGray,fGray,1.0);
        }
    }
}
