#version 430

#define BUFFER_BINDING_VISIBLE_LABEL_ARRAY 7

layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_ARRAY) buffer VisibleLabelArray
{
    int visibleLabelArray[];
};

uniform int iVisibleLabelCount;

float fMaskActiveThreshold = 0.5f;

//计算当前采样位置点附近8整数点的体素值及权重
void CalculateNeighborVoxelInfo(sampler3D s3DTexture, vec3 v3TexCoord, out float fNeighborValues[8], out float fNeighborWeights[8])
{
    vec3 v3DataDim = textureSize(s3DTexture, 0);
    vec3 v3ReciprocalDataDim = 1.0f / v3DataDim;

    vec3 vSamplePosition = v3DataDim * v3TexCoord;
    vec3 vGridPosition = floor(vSamplePosition);

    vec3 vLBN = (vGridPosition + vec3(0.0f, 0.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;
    vec3 vLBF = (vGridPosition + vec3(0.0f, 0.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;

    vec3 vLTN = (vGridPosition + vec3(0.0f, 1.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;
    vec3 vLTF = (vGridPosition + vec3(0.0f, 1.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f))* v3ReciprocalDataDim;

    vec3 vRTN = (vGridPosition + vec3(1.0f, 1.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;
    vec3 vRTF = (vGridPosition + vec3(1.0f, 1.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;

    vec3 vRBN = (vGridPosition + vec3(1.0f, 0.0f, 0.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;
    vec3 vRBF = (vGridPosition + vec3(1.0f, 0.0f, 1.0f) + vec3(0.5f, 0.5f, 0.5f)) * v3ReciprocalDataDim;

    fNeighborValues[0] = texture(s3DTexture, vLBN).x * 255.0f;
    fNeighborValues[1] = texture(s3DTexture, vLBF).x * 255.0f;

    fNeighborValues[2] = texture(s3DTexture, vLTN).x * 255.0f;
    fNeighborValues[3] = texture(s3DTexture, vLTF).x * 255.0f;

    fNeighborValues[4] = texture(s3DTexture, vRTN).x * 255.0f;
    fNeighborValues[5] = texture(s3DTexture, vRTF).x * 255.0f;

    fNeighborValues[6] = texture(s3DTexture, vRBN).x * 255.0f;
    fNeighborValues[7] = texture(s3DTexture, vRBF).x * 255.0f;

    vec3 v3SamplePosFraction = fract(vSamplePosition);
    vec3 v3InverseSamplePosFraction = vec3(1.0f, 1.0f, 1.0f) - v3SamplePosFraction;

    fNeighborWeights[0] = v3InverseSamplePosFraction.x * v3InverseSamplePosFraction.y * v3InverseSamplePosFraction.z;
    fNeighborWeights[1] = v3InverseSamplePosFraction.x * v3InverseSamplePosFraction.y * v3SamplePosFraction.z;

    fNeighborWeights[2] = v3InverseSamplePosFraction.x * v3SamplePosFraction.y * v3InverseSamplePosFraction.z;
    fNeighborWeights[3] = v3InverseSamplePosFraction.x * v3SamplePosFraction.y * v3SamplePosFraction.z;

    fNeighborWeights[4] = v3SamplePosFraction.x * v3SamplePosFraction.y * v3InverseSamplePosFraction.z;
    fNeighborWeights[5] = v3SamplePosFraction.x * v3SamplePosFraction.y * v3SamplePosFraction.z;

    fNeighborWeights[6] = v3SamplePosFraction.x * v3InverseSamplePosFraction.y * v3InverseSamplePosFraction.z;
    fNeighborWeights[7] = v3SamplePosFraction.x * v3InverseSamplePosFraction.y * v3SamplePosFraction.z;
}

//邻域mask二值化后线性插值
float LinearMask(float fActiveLabel, float fNeighborMaskLabels[8], float fNeighborMaskWeights[8])
{
    //邻域label号的二值化
    float fNeighborMaskBinarizedLabels[8];
    for(int i = 0; i < 8; ++i)
    {
        fNeighborMaskBinarizedLabels[i] = abs(fNeighborMaskLabels[i] - fActiveLabel) < fMaskActiveThreshold ? 1.0f : 0.0f;
    }

    //邻域二值化后加权平均
    float fLinearLabel = 0;
    for(int i = 0; i < 8; ++i)
    {
        fLinearLabel += fNeighborMaskBinarizedLabels[i] * fNeighborMaskWeights[i];
    }

    return fLinearLabel;
}

bool AccessMask(sampler3D sampler , vec3 vPos, out int iOutLabel)
{
    //1.计算当前采样位置点最近邻八整数点的mask label值及对采样点的权重
    float fMaskNeighborLabel[8];
    float fMaskNeighborWeight[8];
    CalculateNeighborVoxelInfo(sampler, vPos, fMaskNeighborLabel, fMaskNeighborWeight);

    //2.根据输入Visible label array里label号及优先级返回该位置label号索引及是否可见
    float fActiveMaskLabel = 0.0f;
    float fLinearMaskLabel = 0.0f;

    for(int i= 0 ; i<iVisibleLabelCount ; ++i)
    {
        fActiveMaskLabel = float(visibleLabelArray[i]);

        // label = 0 means air
        if(fActiveMaskLabel < fMaskActiveThreshold)
        {
            return false;
        }

        //根据当前active label对采样点label号做线性插值
        //遍历所有active label时找到第一个符合条件的线性插值结果即返回
        fLinearMaskLabel  = LinearMask(fActiveMaskLabel, fMaskNeighborLabel, fMaskNeighborWeight);
        if(fLinearMaskLabel >= fMaskActiveThreshold)
        {
            iOutLabel = visibleLabelArray[i];
            return true;
        }
    }
    return false;
}
