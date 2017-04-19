#version 430

#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 5

uniform sampler1D sPseudoColor;
uniform float fPseudoColorSlope;
uniform float fPseudoColorIntercept;

uniform vec2 vGlobalWL;

void Composite(vec3 samplePosVolume, vec3 vRayDir, in out vec4 vIntegralColor,
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);

void ColorInverse(in out float fGray);

vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift , int iRayCastStepCode)
{
    float fSumGray = 0.0;
    float fSumNum = 0.0;

    if(0 != (iRayCastStepCode & 0x0001))//First sub data
    {
        fSumGray = 0.0;
        fSumNum = 0.0;
    }

    if(0 != (iRayCastStepCode & 0x0002))//Middle sub data
    {
        fSumGray = vIntegralColor.r;
        fSumNum = vIntegralColor.g;
    }

    vec3 vSamplePos;
    vec4 vCurIntegralColor =  vIntegralColor;

    for (float i = fStartStep ; i <= fEndStep ; ++i)
    {
        vSamplePos = vRayStart + vRayDir * i;
        Composite(vSamplePos , vRayDir, vCurIntegralColor , sVolume , sMask , vSubDataDim ,vSubDataOffset ,vSampleShift );
        fSumGray += vCurIntegralColor.r*100.0;
        ++fSumNum;
    }

    vCurIntegralColor = vec4(fSumGray , fSumNum , 0 ,0);

    //Last sub data transfer gray to color
    if(0 != (iRayCastStepCode & 0x0004))
    {
        float fWW = vGlobalWL.x;
        float fWL = vGlobalWL.y;
        float fResultGray = fSumGray/fSumNum/100.0;
        float fMinGray = fWL - 0.5 * fWW;
        fResultGray= (fResultGray - fMinGray) / fWW;
        fResultGray = clamp(fResultGray, 0.0, 1.0);
        if(fResultGray < 0.000001)
        {
            discard;
        }

        ColorInverse(fResultGray);
        vCurIntegralColor = vec4(texture(sPseudoColor, (fResultGray*fPseudoColorSlope + fPseudoColorIntercept)).rgb, 1.0);
    }

    return vCurIntegralColor;

}