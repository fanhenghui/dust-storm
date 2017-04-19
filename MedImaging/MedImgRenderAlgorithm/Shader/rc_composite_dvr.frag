#version 430
#extension GL_EXT_texture_array : enable

#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 5

uniform sampler1DArray sColorTableArray; 
uniform float fColorTableTexShift;
uniform float fOpacityCompensation; 
uniform float fSampleRate;

//Window level buffer
layout (std430 , binding = BUFFER_BINDING_WINDOW_LEVEL_BUCKET) buffer WindowLevelBucket
{
    vec2 windowing[];
};

bool AccessMask(vec3 vPos, sampler3D sampler , out int iOutLabel);
float AccessVolume(sampler3D sampler , vec3 vPos);

vec4 Shade(vec3 vSamplePos, vec4 vOutputColor, vec3 vRayDir , sampler3D sampler , vec3 vPosInVolume , vec3 vSampleShift , int iLabel);

void Composite(vec3 vSamplePos,vec3 vRayDir, in out vec4 vIntegralColor, 
sampler3D sVolume , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift)
{
    int iLabel = 0;
    vec4 vCurColor = vec4(0.0,0.0,0.0,0.0);
    float fMinGray;
    float fGray;

    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset )/vSubDataDim;//Actual SamplePos in sampler
    if(AccessMask(sMask , vActualSamplePos , iLabel))
    {
        fMinGray = windowing[iLabel].y - 0.5 * windowing[iLabel].x;

        fGray = AccessVolume(sVolume, vActualSamplePos);
        fGray = (fGray - fMinGray) / windowing[iLabel].x;

        vCurColor = texture1DArray(sColorTableArray, vec2(fGray + fColorTableTexShift , iLabel) );
        if(vCurColor.a >0.0)
        {
            vec4 vShading = Shade(vActualSamplePos, vCurColor, vRayDir , sVolume , vSamplePos , vSampleShift , iLabel);
            vShading.a = 1 - pow(1 - vShading.a, fSampleRate/fOpacityCompensation);
            vIntegralColor.rgb += vShading.rgb * (1.0 - vIntegralColor.a) * vShading.a;
            vIntegralColor.a += vShading.a * (1.0 - vIntegralColor.a);
         }
    }
}
