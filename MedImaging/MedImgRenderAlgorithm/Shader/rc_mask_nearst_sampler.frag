#version 430

#define BUFFER_BINDING_VISIBLE_LABEL_BUCKET 6

layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_BUCKET) buffer VisibleLabelBucket
{
    int visibleLabel[];
};

bool access_mask(sampler3D sampler , vec3 pos , out int out_label)
{
    out_label = int(texture(sampler, pos).r*255);
    if(0 == out_label)//0 is invisible
    {
        return false;
    }

    if(1 == visibleLabel[out_label])
    {
        return true;
    }
    else
    {
        return false;
    }
}