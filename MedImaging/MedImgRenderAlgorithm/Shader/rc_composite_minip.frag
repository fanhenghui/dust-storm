#version 430

float fGlobalMinGray;

uniform float fCustomMinThreshold;

bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel);
float AccessVolume(sampler3D sampler , vec3 vPos);

void Composite(vec3 vSamplePos, vec3 vRayDir, in out vec4 vIntegralColor,
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift)
{
    int iLabel = 0;
    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset)/vSubDataDim;
    if(AccessMask(sMask , vActualSamplePos , iLabel))
    {
        float fGray = AccessVolume(sVolume, vActualSamplePos);
        if (fGray > fCustomMinThreshold && fGray < fGlobalMinGray)
        { 
            fGlobalMinGray  = fGray;
            vIntegralColor = vec4(fGray,fGray,fGray,1.0);
        }
    }
}
