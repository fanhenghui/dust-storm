#version 430

#define BUFFER_BINDING_VISIBLE_LABEL_BUCKET 1
#define BUFFER_BINDING_VISIBLE_LABEL_ARRAY 2

layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_BUCKET) buffer VisibleLabelBucket
{
    int visibleLabel[];
};

layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_ARRAY) buffer VisibleLabelArray
{
    int visibleLabelArray[];
};
uniform int visible_label_count;


float label_neighbour[8];
float label_weight[8];
void get_neighbour_info(sampler3D sampler , vec3 pos)
{
    vec3 dim = textureSize(sampler, 0);
    vec3 dim_r = vec3(1.0,1.0,1.0)/dim;
    vec3 sample_pos = dim * pos - vec3(0.5,0.5,0.5);
    vec3 pos_origin = floor(sample_pos);

    vec3 pos_000 = (pos_origin + vec3(0.0f, 0.0f, 0.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_001 = (pos_origin + vec3(0.0f, 0.0f, 1.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_010 = (pos_origin + vec3(0.0f, 1.0f, 0.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_011 = (pos_origin + vec3(0.0f, 1.0f, 1.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_110 = (pos_origin + vec3(1.0f, 1.0f, 0.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_111 = (pos_origin + vec3(1.0f, 1.0f, 1.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_100 = (pos_origin + vec3(1.0f, 0.0f, 0.0f) + vec3(0.5,0.5,0.5)) * dim_r;
    vec3 pos_101 = (pos_origin + vec3(1.0f, 0.0f, 1.0f) + vec3(0.5,0.5,0.5)) * dim_r;

    label_neighbour[0] = texture(sampler, pos_000).x * 255.0f;
    label_neighbour[1] = texture(sampler, pos_001).x * 255.0f;
    label_neighbour[2] = texture(sampler, pos_010).x * 255.0f;
    label_neighbour[3] = texture(sampler, pos_011).x * 255.0f;

    label_neighbour[4] = texture(sampler, pos_100).x * 255.0f;
    label_neighbour[5] = texture(sampler, pos_101).x * 255.0f;
    label_neighbour[6] = texture(sampler, pos_110).x * 255.0f;
    label_neighbour[7] = texture(sampler, pos_111).x * 255.0f;

    vec3 min_pos = fract(sample_pos);
    vec3 max_pos = vec3(1.0,1.0,1.0) - min_pos;

    label_weight[0] = max_pos.x * max_pos.y * max_pos.z;
    label_weight[1] = max_pos.x * max_pos.y * min_pos.z;
    label_weight[2] = max_pos.x * min_pos.y * max_pos.z;
    label_weight[3] = max_pos.x * min_pos.y * min_pos.z;
    label_weight[4] = min_pos.x * max_pos.y * max_pos.z;
    label_weight[5] = min_pos.x * max_pos.y * min_pos.z;
    label_weight[6] = min_pos.x * min_pos.y * max_pos.z;
    label_weight[7] = min_pos.x * min_pos.y * min_pos.z;
}

float get_weight_sum(float label)
{
    float w = 0;
    for(int i = 0; i<8 ; ++i)
    {
        w += abs(label_neighbour[i] - label) < 0.5 ? label_weight[i] : 0;
    }
    return w;
}

int get_max_weight_sum_label()
{
    float label = float(visibleLabelArray[0]);
    float ws = get_weight_sum(label);
    float max_label = label;
    float max_ws = ws;
    for(int i = 1; i < visible_label_count; ++i)
    {
        float label = float(visibleLabelArray[i]);
        float ws = get_weight_sum(label);
        if(ws > max_ws) 
        {
            max_label = label;
            max_ws = ws;
        }
    }

    return int(max_label);
}

bool access_mask(sampler3D sampler , vec3 pos , out int out_label)
{
    float nearst_label = texture(sampler, pos).r*255;
    //check isotropic

    if(0 == nearst_label)//0 is invisible
    {
        return false;
    }

    get_neighbour_info(sampler , pos);
    out_label = get_max_weight_sum_label();

    if(0 != out_label)
    {
        return true;
    }
    else
    {
        return false;
    }
}