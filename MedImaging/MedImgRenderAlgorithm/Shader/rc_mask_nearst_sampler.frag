#version 430

#define BUFFER_BINDING_VISIBLE_LABEL_BUCKET 6

layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_BUCKET) buffer VisibleLabelBucket
{
    int visibleLabel[];
};

bool AccessMask(sampler3D sampler , vec3 vPos , out int iOutLabel)
{
    iOutLabel = int(texture(sampler, vPos).r*255);
    if(0 == iOutLabel)//0 is invisible
    {
        return false;
    }

    if(1 == visibleLabel[iOutLabel])
    {
        return true;
    }
    else
    {
        return false;
    }
}