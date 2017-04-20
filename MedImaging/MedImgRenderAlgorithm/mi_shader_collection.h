#pragma  once

////MPR entry exit points
static const char* ksMPREntryExitPointsComp = "\
#version 430\n\
\n\
#define IMAGE_ENTRY_POINT 0\n\
#define IMAGE_EXIT_POINT 1\n\
#define DISPLAY_SIZE 2\n\
#define VOLUME_DIM 3\n\
#define MVP_INVERSE 4\n\
#define THICKNESS 5\n\
#define RAY_DIRECTION 6\n\
\n\
layout (local_size_x = 4 , local_size_y = 4) in;\n\
\n\
layout (binding = IMAGE_ENTRY_POINT, rgba32f) uniform image2D imgEntryPoints;\n\
layout (binding = IMAGE_EXIT_POINT, rgba32f) uniform image2D imgExitPoints;\n\
\n\
layout (location = DISPLAY_SIZE) uniform uvec2 vDisplaySize;\n\
layout (location = VOLUME_DIM) uniform vec3 vVolumeDim;\n\
layout (location = MVP_INVERSE) uniform mat4 matMVPInverse;\n\
layout (location = THICKNESS) uniform float fThickness;\n\
layout (location = RAY_DIRECTION) uniform vec3 vRayDir;\n\
\n\
/// Get exaclty point\n\
float RayBrickIntersectInit(vec3 initialPt, vec3 brickMin, vec3 brickDim, vec3 rayDir, \n\
    out float startStep, out float endStep)\n\
{\n\
    vec3 invR = 1.0 / (rayDir); \n\
\n\
    vec3 vecBot =  (brickMin - initialPt);\n\
    vec3 vecTop =  (brickMin + brickDim - initialPt);\n\
    vec3 tbot = invR * vecBot;\n\
    vec3 ttop = invR * vecTop; \n\
\n\
    vec3 tmin = min(tbot, ttop);\n\
    vec3 tmax = max(tbot, ttop);\n\
    float tnear = max(max(tmin.x,tmin.y),tmin.z);\n\
    float tfar = min(min(tmax.x,tmax.y),tmax.z);\n\
\n\
    startStep = tnear;   \n\
    startStep = max(startStep, 0.0);\n\
    endStep = tfar;\n\
\n\
    return tnear - startStep;\n\
}\n\
\n\
bool CheckOutside(vec3 point, vec3 boundary)\n\
{\n\
    bvec3 bCompareMin = lessThan(point, vec3(0.0, 0.0, 0.0));\n\
    bvec3 bCompareMax = greaterThan(point, boundary);\n\
    return any(bCompareMin) || any(bCompareMax);\n\
}\n\
\n\
void main()\n\
{\n\
    const ivec2 vImgCoord = ivec2(gl_GlobalInvocationID.xy);\n\
    if(vImgCoord.x > vDisplaySize.x -1  || vImgCoord.y > vDisplaySize.y -1)\n\
    {\n\
        return;\n\
    }\n\
\n\
    //imageStore(imgEntryPoints , vImgCoord , vec4(vImgCoord.x,vImgCoord.y,100, 255));\n\
    //return;\n\
\n\
    float x = (float(vImgCoord.x) +0.5)/float(vDisplaySize.x);\n\
    float y = (float(vImgCoord.y) +0.5)/float(vDisplaySize.y);\n\
\n\
    vec3 vPosNDC = vec3(x*2.0-1.0 , y*2.0-1.0 , 0.0);//not DC to NDC , just NDC to memory\n\
\n\
    vec4 vCentral4 = matMVPInverse * vec4(vPosNDC,1.0);\n\
    vec3 vCentral = vCentral4.xyz / vCentral4.w;\n\
\n\
    vec3 vEntry = vCentral - vRayDir * fThickness *0.5 ;\n\
    vec3 vExit  = vCentral + vRayDir * fThickness * 0.5 ;\n\
\n\
    //imageStore(imgEntryPoints , vImgCoord , vec4(vEntry, 1.0f));\n\
    //imageStore(imgExitPoints , vImgCoord , vec4(vExit, 1.0f));\n\
    //return;\n\
\n\
    float fEntryStep = 0.0;\n\
    float fExitStep = 0.0;\n\
\n\
    vec3 vEntryIntersection = vEntry;\n\
    vec3 vExitIntersection = vExit;\n\
\n\
    RayBrickIntersectInit(vEntry, vec3(0,0,0),vVolumeDim, vRayDir, fEntryStep, fExitStep);\n\
\n\
    //Entry point outside\n\
    if( CheckOutside(vEntry, vVolumeDim) )\n\
    {\n\
        if(fEntryStep >= fExitStep || fEntryStep < 0 || fEntryStep > fThickness)// check entry points in range of thickness and volume\n\
        {\n\
            fExitStep = -1.0;\n\
            imageStore(imgEntryPoints , vImgCoord , vec4(0,0,0, -1.0f));\n\
            imageStore(imgExitPoints , vImgCoord , vec4(0,0,0, -1.0f));\n\
            return;\n\
        }\n\
        vEntryIntersection = vEntry + fEntryStep * vRayDir;\n\
    }\n\
\n\
    //Exit point outside\n\
    if( CheckOutside(vExit, vVolumeDim) )\n\
    {\n\
        if(fEntryStep >= fExitStep)\n\
        {\n\
            fExitStep = -1.0;\n\
            imageStore(imgEntryPoints , vImgCoord , vec4(0,0,0, -1.0f));\n\
            imageStore(imgExitPoints , vImgCoord , vec4(0,0,0, -1.0f));\n\
            return;\n\
        }\n\
        vExitIntersection= vEntry + fExitStep * vRayDir;\n\
    }\n\
\n\
    imageStore(imgEntryPoints , vImgCoord , vec4(vEntryIntersection, 1.0f));\n\
    imageStore(imgExitPoints , vImgCoord , vec4(vExitIntersection, 1.0f));\n\
\n\
    //imageStore(imgEntryPoints , vImgCoord , vec4(255,0,0, 1.0f));\n\
    //imageStore(imgExitPoints , vImgCoord , vec4(vExitIntersection, 1.0f));\n\
\n\
}\n\
";

////GPU ray caster
////1 Main
static const char* ksRCMainVert = "\
#version 430\n\
\n\
layout (location = 0) in vec4 vVertex;\n\
//layout (location = 1) in vec4 vColor;\n\
\n\
void main()\n\
{\n\
    gl_Position = vec4(vVertex.xy,0.0,1.0);\n\
}\n\
";

static const char* ksRCMainFrag = "\
#version 430\n\
\n\
layout (location = 0) out vec4 oFragColor;\n\
\n\
#define IMG_BINDING_ENTRY_POINTS  0\n\
#define IMG_BINDING_EXIT_POINTS  1\n\
\n\
layout (binding = IMG_BINDING_ENTRY_POINTS, rgba32f) readonly uniform image2D imgEntryPoints;\n\
layout (binding = IMG_BINDING_EXIT_POINTS, rgba32f) readonly uniform image2D imgExitPoints;\n\
\n\
uniform vec3 vVolumeDim;\n\
uniform sampler3D sVolume;\n\
uniform sampler3D sMask;\n\
uniform float fSampleRate;\n\
\n\
void Preprocess(out vec3 vRayStart,out vec3 vRayDirWithSampleRate, out float fStartStep, out float fEndStep)\n\
{\n\
    ivec2 vFragCoord = ivec2(gl_FragCoord.x, gl_FragCoord.y);\n\
    vec3 vStartPoint = imageLoad(imgEntryPoints, vFragCoord.xy).xyz;\n\
    vec3 vEndPoint = imageLoad(imgExitPoints, vFragCoord.xy).xyz;\n\
\n\
    vec3 vRayDir = vEndPoint - vStartPoint;\n\
    vec3 vRayDirNorm = normalize(vRayDir);\n\
    float fRayLength = length(vRayDir);\n\
\n\
    if(fRayLength < 1e-5)\n\
    {\n\
        discard;\n\
    }\n\
\n\
    vRayStart = vStartPoint;\n\
    vRayDirWithSampleRate = vRayDirNorm* fSampleRate;\n\
    fStartStep = 0;\n\
    fEndStep = fRayLength/ fSampleRate;\n\
}\n\
\n\
//Ray cast step code : \n\
//1 first sub data step \n\
//2 middle sub data step \n\
//4 last sub data step\n\
vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,\n\
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift , int iRayCastStepCode);\n\
\n\
void main()\n\
{\n\
    vec3 vRayStrat = vec3(0,0,0);\n\
    vec3 vRayDirWithSampleRate = vec3(1,0,0);\n\
    float fEndStep = 0;\n\
    float fStartStep = 0;\n\
\n\
    vec4 vIntegralColor = vec4(0,0,0,0);\n\
\n\
    Preprocess(vRayStrat, vRayDirWithSampleRate, fStartStep, fEndStep);\n\
\n\
    oFragColor = Raycast(\n\
        vRayStrat, \n\
        vRayDirWithSampleRate, \n\
        fStartStep, \n\
        fEndStep, \n\
        vIntegralColor , \n\
        sVolume , \n\
        sMask , \n\
        vVolumeDim,\n\
        vec3(0.0), \n\
        vec3(1.0)/vVolumeDim,\n\
        5);\n\
}\n\
";

////2 Ray casting
static const char* ksRCRayCastingDVRFrag = "\
#version 430\n\
\n\
bool CheckOpacity(in out float opacity);\n\
\n\
void Composite(vec3 samplePosVolume, vec3 vRayDir, in out vec4 vIntegralColor,\n\
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);\n\
\n\
vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,\n\
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift, int iRayCastStepCode)\n\
{\n\
\n\
}\n\
";

static const char* ksRCRayCastingMIPMinIPFrag = "\
#version 430\n\
\n\
uniform sampler1D sPseudoColor;\n\
uniform float fPseudoColorSlope;\n\
uniform float fPseudoColorIntercept;\n\
\n\
uniform vec2 vGlobalWL;\n\
\n\
float fGlobalMaxGray = -65535.0;\n\
float fGlobalMinGray = 65535.0;\n\
\n\
void Composite(vec3 samplePosVolume, vec3 vRayDir, in out vec4 vIntegralColor,\n\
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);\n\
\n\
void ColorInverse(in out float fGray);\n\
\n\
vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,\n\
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift)\n\
{\n\
    if(0!= (iRayCastStepCode & 0x0001))//First sub data\n\
    {\n\
        fGlobalMaxGray = -65535.0;\n\
        fGlobalMinGray = 65535.0;\n\
    }\n\
\n\
    if(0!= (iRayCastStepCode & 0x0002))//Middle sub data\n\
    {\n\
        fGlobalMaxGray = vIntegralColor.r;\n\
        fGlobalMinGray = vIntegralColor.r;\n\
    }\n\
\n\
    vec3 vSamplePos;\n\
    vec4 vCurIntegralColor =  vIntegralColor;\n\
\n\
    for (float i = fStartStep ; i <= fEndStep ; ++i)\n\
    {\n\
        vSamplePos = vRayStart + vRayDir * i;\n\
        Composite(vSamplePos , vRayDir, vCurIntegralColor , sVolume , sMask , vSubDataDim , vSubDataOffset ,vSampleShift );\n\
    }\n\
\n\
    //Last sub data transfer gray to color\n\
    if(0!= (iRayCastStepCode & 0x0004))\n\
    {\n\
        float fWW = vGlobalWL.x;\n\
        float fWL = vGlobalWL.y;\n\
        float fResultGray = vCurIntegralColor.r;\n\
        float fMinGray = fWL - 0.5 * fWW;\n\
        fResultGray= (fResultGray - fMinGray) / fWW;\n\
        fResultGray = clamp(fResultGray, 0.0, 1.0);\n\
        if(fResultGray < 0.000001)\n\
        {\n\
            discard;\n\
        }\n\
\n\
        ColorInverse(fResultGray);\n\
        vCurIntegralColor = vec4(texture(sPseudoColor, (fResultGray*fPseudoColorSlope + fPseudoColorIntercept)).rgb, 1.0);\n\
    }\n\
\n\
    return vCurIntegralColor;\n\
}\n\
";

static const char* ksRCRayCastingAverageFrag = "\
#version 430\n\
\n\
#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 5\n\
\n\
uniform sampler1D sPseudoColor;\n\
uniform float fPseudoColorSlope;\n\
uniform float fPseudoColorIntercept;\n\
\n\
uniform vec2 vGlobalWL;\n\
\n\
void Composite(vec3 samplePosVolume, vec3 vRayDir, in out vec4 vIntegralColor,\n\
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift);\n\
\n\
void ColorInverse(in out float fGray);\n\
\n\
vec4 Raycast(vec3 vRayStart, vec3 vRayDir, float fStartStep, float fEndStep, vec4 vIntegralColor,\n\
    sampler3D sVolume,  sampler3D sMask,   vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift , int iRayCastStepCode)\n\
{\n\
    float fSumGray = 0.0;\n\
    float fSumNum = 0.0;\n\
\n\
    if(0 != (iRayCastStepCode & 0x0001))//First sub data\n\
    {\n\
        fSumGray = 0.0;\n\
        fSumNum = 0.0;\n\
    }\n\
\n\
    if(0 != (iRayCastStepCode & 0x0002))//Middle sub data\n\
    {\n\
        fSumGray = vIntegralColor.r;\n\
        fSumNum = vIntegralColor.g;\n\
    }\n\
\n\
    vec3 vSamplePos;\n\
    vec4 vCurIntegralColor =  vIntegralColor;\n\
\n\
    for (float i = fStartStep ; i <= fEndStep ; ++i)\n\
    {\n\
        vSamplePos = vRayStart + vRayDir * i;\n\
        Composite(vSamplePos , vRayDir, vCurIntegralColor , sVolume , sMask , vSubDataDim ,vSubDataOffset ,vSampleShift );\n\
        fSumGray += vCurIntegralColor.r*100.0;\n\
        ++fSumNum;\n\
    }\n\
\n\
    vCurIntegralColor = vec4(fSumGray , fSumNum , 0 ,0);\n\
\n\
    //Last sub data transfer gray to color\n\
    if(0 != (iRayCastStepCode & 0x0004))\n\
    {\n\
        float fWW = vGlobalWL.x;\n\
        float fWL = vGlobalWL.y;\n\
        float fResultGray = fSumGray/fSumNum/100.0;\n\
        float fMinGray = fWL - 0.5 * fWW;\n\
        fResultGray= (fResultGray - fMinGray) / fWW;\n\
        fResultGray = clamp(fResultGray, 0.0, 1.0);\n\
        if(fResultGray < 0.000001)\n\
        {\n\
            discard;\n\
        }\n\
\n\
        ColorInverse(fResultGray);\n\
        vCurIntegralColor = vec4(texture(sPseudoColor, (fResultGray*fPseudoColorSlope + fPseudoColorIntercept)).rgb, 1.0);\n\
    }\n\
\n\
    return vCurIntegralColor;\n\
\n\
}\n\
";

////3 Composite
static const char* ksRCCompositeAverageFrag = "\
#version 430\n\
\n\
bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel);\n\
float AccessVolume(sampler3D sampler , vec3 vPos);\n\
\n\
void Composite(vec3 vSamplePos, vec3 vRayDir, in out vec4 vIntegralColor,\n\
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift)\n\
{\n\
    int iLabel = 0;\n\
    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset)/vSubDataDim;\n\
    if(AccessMask(sMask , vActualSamplePos , iLabel))\n\
    {\n\
        float fGray = AccessVolume(sVolume, vActualSamplePos);\n\
        vIntegralColor = vec4(fGray,fGray,fGray,1.0);\n\
    }\n\
}\n\
";

static const char* ksRCCompositeDVRFrag = "\
#version 430\n\
#extension GL_EXT_texture_array : enable\n\
\n\
#define BUFFER_BINDING_WINDOW_LEVEL_BUCKET 5\n\
\n\
uniform sampler1DArray sColorTableArray; \n\
uniform float fColorTableTexShift;\n\
uniform float fOpacityCompensation; \n\
uniform float fSampleRate;\n\
\n\
//Window level buffer\n\
layout (std430 , binding = BUFFER_BINDING_WINDOW_LEVEL_BUCKET) buffer WindowLevelBucket\n\
{\n\
    vec2 windowing[];\n\
};\n\
\n\
bool AccessMask(vec3 vPos, sampler3D sampler , out int iOutLabel);\n\
float AccessVolume(sampler3D sampler , vec3 vPos);\n\
\n\
vec4 Shade(vec3 vSamplePos, vec4 vOutputColor, vec3 vRayDir , sampler3D sampler , vec3 vPosInVolume , vec3 vSampleShift , int iLabel);\n\
\n\
void Composite(vec3 vSamplePos,vec3 vRayDir, in out vec4 vIntegralColor, \n\
sampler3D sVolume , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset , vec3 vSampleShift)\n\
{\n\
    int iLabel = 0;\n\
    vec4 vCurColor = vec4(0.0,0.0,0.0,0.0);\n\
    float fMinGray;\n\
    float fGray;\n\
\n\
    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset )/vSubDataDim;//Actual SamplePos in sampler\n\
    if(AccessMask(sMask , vActualSamplePos , iLabel))\n\
    {\n\
        fMinGray = windowing[iLabel].y - 0.5 * windowing[iLabel].x;\n\
\n\
        fGray = AccessVolume(sVolume, vActualSamplePos);\n\
        fGray = (fGray - fMinGray) / windowing[iLabel].x;\n\
\n\
        vCurColor = texture1DArray(sColorTableArray, vec2(fGray + fColorTableTexShift , iLabel) );\n\
        if(vCurColor.a >0.0)\n\
        {\n\
            vec4 vShading = Shade(vActualSamplePos, vCurColor, vRayDir , sVolume , vSamplePos , vSampleShift , iLabel);\n\
            vShading.a = 1 - pow(1 - vShading.a, fSampleRate/fOpacityCompensation);\n\
            vIntegralColor.rgb += vShading.rgb * (1.0 - vIntegralColor.a) * vShading.a;\n\
            vIntegralColor.a += vShading.a * (1.0 - vIntegralColor.a);\n\
         }\n\
    }\n\
}\n\
";

static const char* ksRCCompositeMIPFrag = "\
#version 430\n\
\n\
float fGlobalMaxGray;\n\
bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel);\n\
float AccessVolume(sampler3D sampler , vec3 vPos);\n\
\n\
void Composite(vec3 vSamplePos, vec3 vRayDir, in out vec4 vIntegralColor,\n\
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift)\n\
{\n\
    int iLabel = 0;\n\
    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset)/vSubDataDim;\n\
    if(AccessMask(sMask , vActualSamplePos , iLabel))\n\
    {\n\
        float fGray = AccessVolume(sVolume, vActualSamplePos);\n\
        if(fGlobalMaxGray < fGray)\n\
        {\n\
            fGlobalMaxGray = fGray;\n\
            vIntegralColor = vec4(fGray,fGray,fGray,1.0);\n\
        }\n\
    }\n\
}\n\
";

static const char* ksRCCompositeMinIPFrag = "\
#version 430\n\
\n\
float fGlobalMinGray;\n\
\n\
uniform float fCustomMinThreshold;\n\
\n\
bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel);\n\
float AccessVolume(sampler3D sampler , vec3 vPos);\n\
\n\
void Composite(vec3 vSamplePos, vec3 vRayDir, in out vec4 vIntegralColor,\n\
    sampler3D sVolume  , sampler3D sMask , vec3 vSubDataDim , vec3 vSubDataOffset,  vec3 vSampleShift)\n\
{\n\
    int iLabel = 0;\n\
    vec3 vActualSamplePos = (vSamplePos + vSubDataOffset)/vSubDataDim;\n\
    if(AccessMask(sMask , vActualSamplePos , iLabel))\n\
    {\n\
        float fGray = AccessVolume(sVolume, vActualSamplePos);\n\
        if (fGray > fCustomMinThreshold && fGray < fGlobalMinGray)\n\
        { \n\
            fGlobalMinGray  = fGray;\n\
            vIntegralColor = vec4(fGray,fGray,fGray,1.0);\n\
        }\n\
    }\n\
}\n\
";

////4 Shading
static const char* ksRCShadingNoneFrag = "\
#version 430\n\
\n\
vec4 Shade(vec3 sampleCoord, vec4 sampleColor, vec3 rayDir , sampler3D vDataVolume , vec3 samplePosVolume , vec3 vSampleShift , int idx)\n\
{\n\
    return sampleColor;\n\
}\n\
";

static const char* ksRCShadingPhongFrag = "\
#version 430\n\
\n\
\n\
vec4 Shade(vec3 vSamplePos, vec4 vOutputColor, vec3 vRayDir , sampler3D sampler , vec3 vPosInVolume , vec3 vSampleShift , int iLabel)\n\
{\n\
    return sampleColor;\n\
}\n\
";

////5 Sampler
static const char* ksRCVolumeLinearSamplerFrag = "\
#version 430\n\
\n\
float AccessVolume(sampler3D sampler , vec3 vPos)\n\
{\n\
    return texture(sampler , vPos).r;\n\
}\n\
";

static const char* ksRCVolumeNearstSamplerFrag = "\
#version 430\n\
\n\
float AccessVolume(sampler3D sampler , vec3 vPos)\n\
{\n\
    vec3 vDim = textureSize(sampler, 0);\n\
    return texturefetch(sampler , ivec3(vPos*vDim) , 0 ).r;\n\
}\n\
";

static const char* ksRCMaskLinearSamplerFrag = "\
#version 430\n\
\n\
#define BUFFER_BINDING_VISIBLE_LABEL_ARRAY 7\n\
\n\
layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_ARRAY) buffer VisibleLabelArray\n\
{\n\
    int visibleLabelArray[];\n\
};\n\
\n\
uniform int iVisibleLabelCount;\n\
\n\
float fMaskActiveThreshold = 0.5f;\n\
\n\
//计算当前采样位置点附近8整数点的体素值及权重\n\
void CalculateNeighborVoxelInfo(sampler3D s3DTexture, vec3 v3TexCoord, out float fNeighborValues[8], out float fNeighborWeights[8])\n\
{\n\
    vec3 v3DataDim = textureSize(s3DTexture, 0);\n\
    vec3 v3ReciprocalDataDim = 1.0f / v3DataDim;\n\
\n\
    vec3 vSamplePosition = v3DataDim * v3TexCoord;\n\
    vec3 vGridPosition = floor(vSamplePosition);\n\
\n\
    vec3 vLBN = (vGridPosition + vec3(0.0f, 0.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
    vec3 vLBF = (vGridPosition + vec3(0.0f, 0.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
\n\
    vec3 vLTN = (vGridPosition + vec3(0.0f, 1.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
    vec3 vLTF = (vGridPosition + vec3(0.0f, 1.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f))* v3ReciprocalDataDim;\n\
\n\
    vec3 vRTN = (vGridPosition + vec3(1.0f, 1.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
    vec3 vRTF = (vGridPosition + vec3(1.0f, 1.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
\n\
    vec3 vRBN = (vGridPosition + vec3(1.0f, 0.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
    vec3 vRBF = (vGridPosition + vec3(1.0f, 0.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;\n\
\n\
    fNeighborValues[0] = texture(s3DTexture, vLBN).x * 255.0f;\n\
    fNeighborValues[1] = texture(s3DTexture, vLBF).x * 255.0f;\n\
\n\
    fNeighborValues[2] = texture(s3DTexture, vLTN).x * 255.0f;\n\
    fNeighborValues[3] = texture(s3DTexture, vLTF).x * 255.0f;\n\
\n\
    fNeighborValues[4] = texture(s3DTexture, vRTN).x * 255.0f;\n\
    fNeighborValues[5] = texture(s3DTexture, vRTF).x * 255.0f;\n\
\n\
    fNeighborValues[6] = texture(s3DTexture, vRBN).x * 255.0f;\n\
    fNeighborValues[7] = texture(s3DTexture, vRBF).x * 255.0f;\n\
\n\
    vec3 v3SamplePosFraction = fract(vSamplePosition);\n\
    vec3 v3InverseSamplePosFraction = vec3(1.0f, 1.0f, 1.0f) - v3SamplePosFraction;\n\
\n\
    fNeighborWeights[0] = v3InverseSamplePosFraction.x * v3InverseSamplePosFraction.y * v3InverseSamplePosFraction.z;\n\
    fNeighborWeights[1] = v3InverseSamplePosFraction.x * v3InverseSamplePosFraction.y * v3SamplePosFraction.z;\n\
\n\
    fNeighborWeights[2] = v3InverseSamplePosFraction.x * v3SamplePosFraction.y * v3InverseSamplePosFraction.z;\n\
    fNeighborWeights[3] = v3InverseSamplePosFraction.x * v3SamplePosFraction.y * v3SamplePosFraction.z;\n\
\n\
    fNeighborWeights[4] = v3SamplePosFraction.x * v3SamplePosFraction.y * v3InverseSamplePosFraction.z;\n\
    fNeighborWeights[5] = v3SamplePosFraction.x * v3SamplePosFraction.y * v3SamplePosFraction.z;\n\
\n\
    fNeighborWeights[6] = v3SamplePosFraction.x * v3InverseSamplePosFraction.y * v3InverseSamplePosFraction.z;\n\
    fNeighborWeights[7] = v3SamplePosFraction.x * v3InverseSamplePosFraction.y * v3SamplePosFraction.z;\n\
}\n\
\n\
//邻域mask二值化后线性插值\n\
float LinearMask(float fActiveLabel, float fNeighborMaskLabels[8], float fNeighborMaskWeights[8])\n\
{\n\
    //邻域label号的二值化\n\
    float fNeighborMaskBinarizedLabels[8];\n\
    for(int i = 0; i < 8; ++i)\n\
    {\n\
        fNeighborMaskBinarizedLabels[i] = abs(fNeighborMaskLabels[i] - fActiveLabel) < fMaskActiveThreshold ? 1.0f : 0.0f;\n\
    }\n\
\n\
    //邻域二值化后加权平均\n\
    float fLinearLabel = 0;\n\
    for(int i = 0; i < 8; ++i)\n\
    {\n\
        fLinearLabel += fNeighborMaskBinarizedLabels[i] * fNeighborMaskWeights[i];\n\
    }\n\
\n\
    return fLinearLabel;\n\
}\n\
\n\
bool AccessMask(sampler3D sampler , vec3 vPos, out int iOutLabel)\n\
{\n\
    //1.计算当前采样位置点最近邻八整数点的mask label值及对采样点的权重\n\
    float fMaskNeighborLabel[8];\n\
    float fMaskNeighborWeight[8];\n\
    CalculateNeighborVoxelInfo(sampler, vPos, fMaskNeighborLabel, fMaskNeighborWeight);\n\
\n\
    //2.根据输入Visible label array里label号及优先级返回该位置label号索引及是否可见\n\
    float fActiveMaskLabel = 0.0f;\n\
    float fLinearMaskLabel = 0.0f;\n\
\n\
    for(int i= 0 ; i<iVisibleLabelCount ; ++i)\n\
    {\n\
        fActiveMaskLabel = float(visibleLabelArray[i]);\n\
\n\
        // label = 0 means air\n\
        if(fActiveMaskLabel < fMaskActiveThreshold)\n\
        {\n\
            return false;\n\
        }\n\
\n\
        //根据当前active label对采样点label号做线性插值\n\
        //遍历所有active label时找到第一个符合条件的线性插值结果即返回\n\
        fLinearMaskLabel  = LinearMask(fActiveMaskLabel, fMaskNeighborLabel, fMaskNeighborWeight);\n\
        if(fLinearMaskLabel >= fMaskActiveThreshold)\n\
        {\n\
            iOutLabel = visibleLabelArray[i];\n\
            return true;\n\
        }\n\
    }\n\
    return false;\n\
}\n\
";

static const char* ksRCMaskNearstSamplerFrag = "\
#version 430\n\
\n\
#define BUFFER_BINDING_VISIBLE_LABEL_BUCKET 6\n\
\n\
layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_BUCKET) buffer VisibleLabelBucket\n\
{\n\
    int visibleLabel[];\n\
};\n\
\n\
bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel)\n\
{\n\
    iOutLabel = int(texture(sampler, vPos).r*255);\n\
    if(0 == iOutLabel)//0 is invisible\n\
    {\n\
        return false;\n\
    }\n\
\n\
    if(1 == visibleLabel[iOutLabel])\n\
    {\n\
        return true;\n\
    }\n\
    else\n\
    {\n\
        return false;\n\
    }\n\
}\n\
";

static const char* ksRCMaskNoneSamplerFrag = "\
#version 430\n\
\n\
bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel)\n\
{\n\
    iOutLabel = 0;\n\
    return true;\n\
}\n\
";

////6 Color Inverse
static const char* ksRCColorInverseDisableFrag = "\
#version 430\n\
\n\
void ColorInverse(in out float fGray)\n\
{\n\
    //Do nothing\n\
}\n\
";

static const char* ksRCColorInverseEnableFrag = "\
#version 430\n\
\n\
void ColorInverse(in out float fGray)\n\
{\n\
    fGray = 1.0 - fGray;\n\
}\n\
";

////7 Utils
static const char* ksRCUtilsFrag = "\
#version 430\n\
\n\
bool CheckOpacity(in out float opacity)\n\
{\n\
    if(opacity > 0.95)\n\
    {\n\
        opacity = 1.0;\n\
        return true;\n\
    }\n\
    else\n\
    {\n\
        return false;\n\
    }\n\
}\n\
\n\
\n\
//Encoding label to intger array 4*32 can contain 0~127 labels\n\
void LabelEncode(int iLabel , in out int maskFlag[4])\n\
{\n\
    if(iLabel < 32)\n\
    {\n\
        maskFlag[0] = maskFlag[0] | ( 1 << iLabel );\n\
    }\n\
    else if(iLabel < 64)\n\
    {\n\
        maskFlag[1] = maskFlag[1] | ( 1 << (iLabel-32) );\n\
    }\n\
    else if(iLabel < 96)\n\
    {\n\
        maskFlag[2] = maskFlag[2] | ( 1 << (iLabel-64) );\n\
    }\n\
    else\n\
    {\n\
        maskFlag[3] = maskFlag[3] | ( 1 << (iLabel-96) );\n\
    }\n\
}\n\
\n\
//Decoding label from intger array 4*32 can contain 0~127 labels\n\
bool LabelDecode(int iLabel , int maskFlag[4])\n\
{\n\
\n\
    bool bHitted = false;\n\
    if(iLabel < 32)\n\
    {\n\
        bHitted = ( ( 1 << iLabel ) & maskFlag[0] ) != 0;\n\
    }\n\
    else if(iLabel < 64)\n\
    {\n\
        bHitted = ( ( 1 << (iLabel - 32) ) & maskFlag[1] ) != 0;\n\
    }\n\
    else if(iLabel < 96)\n\
    {\n\
        bHitted = ( ( 1 << (iLabel - 64) ) & maskFlag[2] ) != 0;\n\
    }\n\
    else\n\
    {\n\
        bHitted = ( ( 1 << (iLabel - 96) ) & maskFlag[3] ) != 0;\n\
    }\n\
    return bHitted;\n\
}\n\
";

