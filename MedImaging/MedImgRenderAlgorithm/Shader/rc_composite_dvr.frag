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

bool access_mask(vec3 vPos, sampler3D sampler , out int iOutLabel);
float access_volume(sampler3D sampler , vec3 vPos);

vec4 shade(vec3 vSamplePos, vec4 vOutputColor, vec3 ray_dir , sampler3D sampler , vec3 vPosInVolume , vec3 vSampleShift , int iLabel);

void composite(vec3 vSamplePos,vec3 ray_dir, in out vec4 vIntegralColor, 
sampler3D sVolume , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift)
{
    int iLabel = 0;
    vec4 vCurColor = vec4(0.0,0.0,0.0,0.0);
    float fMinGray;
    float fGray;

    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset )/vSubDataDim;//Actual SamplePos in sampler
    if(access_mask(sMask , vActualSamplePos , iLabel))
    {
        fMinGray = windowing[iLabel].y - 0.5 * windowing[iLabel].x;

        fGray = access_volume(sVolume, vActualSamplePos);
        fGray = (fGray - fMinGray) / windowing[iLabel].x;

        vCurColor = texture1DArray(sColorTableArray, vec2(fGray + fColorTableTexShift , iLabel) );
        if(vCurColor.a >0.0)
        {
            vec4 vShading = shade(vActualSamplePos, vCurColor, ray_dir , sVolume , vSamplePos , vSampleShift , iLabel);
            vShading.a = 1 - pow(1 - vShading.a, fSampleRate/fOpacityCompensation);
            vIntegralColor.rgb += vShading.rgb * (1.0 - vIntegralColor.a) * vShading.a;
            vIntegralColor.a += vShading.a * (1.0 - vIntegralColor.a);
         }
    }
}
