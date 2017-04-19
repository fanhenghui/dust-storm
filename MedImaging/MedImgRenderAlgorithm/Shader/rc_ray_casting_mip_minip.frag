#version 430

uniform sampler1D sPseudoColor;
uniform float fPseudoColorSlope;
uniform float fPseudoColorIntercept;

uniform vec2 vGlobalWL;

float fGlobalMaxGray = -65535.0;
float fGlobalMinGray = 65535.0;

void Composite(vec3 samplePosVolume, vec3 vRayDir, in out vec4 vIntegralColor,
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);

void ColorInverse(in out float fGray);

vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift)
{
    if(0!= (iRayCastStepCode & 0x0001))//First sub data
    {
        fGlobalMaxGray = -65535.0;
        fGlobalMinGray = 65535.0;
    }

    if(0!= (iRayCastStepCode & 0x0002))//Middle sub data
    {
        fGlobalMaxGray = vIntegralColor.r;
        fGlobalMinGray = vIntegralColor.r;
    }

    vec3 vSamplePos;
    vec4 vCurIntegralColor =  vIntegralColor;

    for (float i = fStartStep ; i <= fEndStep ; ++i)
    {
        vSamplePos = vRayStart + vRayDir * i;
        Composite(vSamplePos , vRayDir, vCurIntegralColor , sVolume , sMask , vSubDataDim , vSubDataOffset ,vSampleShift );
    }

    //Last sub data transfer gray to color
    if(0!= (iRayCastStepCode & 0x0004))
    {
        float fWW = vGlobalWL.x;
        float fWL = vGlobalWL.y;
        float fResultGray = vCurIntegralColor.r;
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
