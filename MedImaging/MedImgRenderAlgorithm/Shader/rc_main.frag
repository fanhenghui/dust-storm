#version 430

layout (location = 0) out vec4 oFragColor;

#define IMG_BINDING_ENTRY_POINTS  0
#define IMG_BINDING_EXIT_POINTS  1

layout (binding = IMG_BINDING_ENTRY_POINTS, rgba32f) readonly uniform image2D image_entry_points;
layout (binding = IMG_BINDING_EXIT_POINTS, rgba32f) readonly uniform image2D image_exit_points;

uniform vec3 volume_dim;
uniform sampler3D sVolume;
uniform sampler3D sMask;
uniform float fSampleRate;

void preprocess(out vec3 vRayStart,out vec3 vRayDirWithSampleRate, out float fStartStep, out float fEndStep)
{
    ivec2 vFragCoord = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    vec3 vStartPoint = imageLoad(image_entry_points, vFragCoord.xy).xyz;
    vec3 vEndPoint = imageLoad(image_exit_points, vFragCoord.xy).xyz;

    vec3 ray_dir = vEndPoint - vStartPoint;
    vec3 vRayDirNorm = normalize(ray_dir);
    float fRayLength = length(ray_dir);

    if(fRayLength < 1e-5)
    {
        discard;
    }

    vRayStart = vStartPoint;
    vRayDirWithSampleRate = vRayDirNorm* fSampleRate;
    fStartStep = 0;
    fEndStep = fRayLength/ fSampleRate;
}

//Ray cast step code : 
//1 first sub data step 
//2 middle sub data step 
//4 last sub data step
vec4 raycast(vec3 vRayStart, vec3 ray_dir, float fStartStep, float fEndStep, vec4 vIntegralColor,
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift , int iRayCastStepCode);

void main()
{
    vec3 vRayStrat = vec3(0,0,0);
    vec3 vRayDirWithSampleRate = vec3(1,0,0);
    float fEndStep = 0;
    float fStartStep = 0;

    vec4 vIntegralColor = vec4(0,0,0,0);

    preprocess(vRayStrat, vRayDirWithSampleRate, fStartStep, fEndStep);

    oFragColor = raycast(
        vRayStrat, 
        vRayDirWithSampleRate, 
        fStartStep, 
        fEndStep, 
        vIntegralColor , 
        sVolume , 
        sMask , 
        volume_dim,
        vec3(0.0), 
        vec3(1.0)/volume_dim,
        5);
}