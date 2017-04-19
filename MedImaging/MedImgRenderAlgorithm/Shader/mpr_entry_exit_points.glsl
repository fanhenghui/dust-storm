#version 430

#define IMAGE_ENTRY_POINT 0
#define IMAGE_EXIT_POINT 1
#define DISPLAY_SIZE 2
#define VOLUME_DIM 3
#define MVP_INVERSE 4
#define THICKNESS 5
#define RAY_DIRECTION 6

layout (local_size_x = 4 , local_size_y = 4) in;

layout (binding = IMAGE_ENTRY_POINT, rgba32f) uniform image2D imgEntryPoints;
layout (binding = IMAGE_EXIT_POINT, rgba32f) uniform image2D imgExitPoints;

layout (location = DISPLAY_SIZE) uniform uvec2 vDisplaySize;
layout (location = VOLUME_DIM) uniform vec3 vVolumeDim;
layout (location = MVP_INVERSE) uniform mat4 matMVPInverse;
layout (location = THICKNESS) uniform float fThickness;
layout (location = RAY_DIRECTION) uniform vec3 vRayDir;

/// Get exaclty point
float RayBrickIntersectInit(vec3 initialPt, vec3 brickMin, vec3 brickDim, vec3 rayDir, 
    out float startStep, out float endStep)
{
    vec3 invR = 1.0 / (rayDir); 

    vec3 vecBot =  (brickMin - initialPt);
    vec3 vecTop =  (brickMin + brickDim - initialPt);
    vec3 tbot = invR * vecBot;
    vec3 ttop = invR * vecTop; 

    vec3 tmin = min(tbot, ttop);
    vec3 tmax = max(tbot, ttop);
    float tnear = max(max(tmin.x,tmin.y),tmin.z);
    float tfar = min(min(tmax.x,tmax.y),tmax.z);

    startStep = tnear;   
    startStep = max(startStep, 0.0);
    endStep = tfar;

    return tnear - startStep;
}

bool CheckOutside(vec3 point, vec3 boundary)
{
    bvec3 bCompareMin = lessThan(point, vec3(0.0, 0.0, 0.0));
    bvec3 bCompareMax = greaterThan(point, boundary);
    return any(bCompareMin) || any(bCompareMax);
}

void main()
{
    const ivec2 vImgCoord = ivec2(gl_GlobalInvocationID.xy);
    if(vImgCoord.x > vDisplaySize.x -1  || vImgCoord.y > vDisplaySize.y -1)
    {
        return;
    }

    //imageStore(imgEntryPoints , vImgCoord , vec4(vImgCoord.x,vImgCoord.y,100, 255));
    //return;

    float x = (float(vImgCoord.x) +0.5)/float(vDisplaySize.x);
    float y = (float(vImgCoord.y) +0.5)/float(vDisplaySize.y);

    vec3 vPosNDC = vec3(x*2.0-1.0 , y*2.0-1.0 , 0.0);//not DC to NDC , just NDC to memory

    vec4 vCentral4 = matMVPInverse * vec4(vPosNDC,1.0);
    vec3 vCentral = vCentral4.xyz / vCentral4.w;

    vec3 vEntry = vCentral - vRayDir * fThickness *0.5 ;
    vec3 vExit  = vCentral + vRayDir * fThickness * 0.5 ;

    //imageStore(imgEntryPoints , vImgCoord , vec4(vEntry, 1.0f));
    //imageStore(imgExitPoints , vImgCoord , vec4(vExit, 1.0f));
    //return;

    float fEntryStep = 0.0;
    float fExitStep = 0.0;

    vec3 vEntryIntersection = vEntry;
    vec3 vExitIntersection = vExit;

    RayBrickIntersectInit(vEntry, vec3(0,0,0),vVolumeDim, vRayDir, fEntryStep, fExitStep);

    //Entry point outside
    if( CheckOutside(vEntry, vVolumeDim) )
    {
        if(fEntryStep >= fExitStep || fEntryStep < 0 || fEntryStep > fThickness)// check entry points in range of thickness and volume
        {
            fExitStep = -1.0;
            imageStore(imgEntryPoints , vImgCoord , vec4(0,0,0, -1.0f));
            imageStore(imgExitPoints , vImgCoord , vec4(0,0,0, -1.0f));
            return;
        }
        vEntryIntersection = vEntry + fEntryStep * vRayDir;
    }

    //Exit point outside
    if( CheckOutside(vExit, vVolumeDim) )
    {
        if(fEntryStep >= fExitStep)
        {
            fExitStep = -1.0;
            imageStore(imgEntryPoints , vImgCoord , vec4(0,0,0, -1.0f));
            imageStore(imgExitPoints , vImgCoord , vec4(0,0,0, -1.0f));
            return;
        }
        vExitIntersection= vEntry + fExitStep * vRayDir;
    }

    imageStore(imgEntryPoints , vImgCoord , vec4(vEntryIntersection, 1.0f));
    imageStore(imgExitPoints , vImgCoord , vec4(vExitIntersection, 1.0f));

    //imageStore(imgEntryPoints , vImgCoord , vec4(255,0,0, 1.0f));
    //imageStore(imgExitPoints , vImgCoord , vec4(vExitIntersection, 1.0f));

}