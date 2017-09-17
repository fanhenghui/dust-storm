#version 430 

#define BRICK_MASK_INFO_BUFFER 0
#define VISIBLE_LABEL_BUFFER 1
#define BRICK_SIZE 2
#define BRICK_MARGIN 3
#define BRICK_DIM 4
#define MASK_DIM 5
#define MASK_TEXTURE 6
#define BRICK_RANGE_MIN 7
#define BRICK_RANGE_DIM 8
#define VISIBLE_LABEL_COUNT 9

//One local work group handle 8x8x1 64 bricks
layout (local_size_x = 8 , local_size_y = 8 , local_size_z = 1) in;

struct BrickMaskInfo
{
    int labelCode;
};
layout (std430 , binding = BRICK_MASK_INFO_BUFFER ) buffer BrickMaskInfoInOut
{
    BrickMaskInfo brickMaskInfo[];
};

layout (std430 , binding = VISIBLE_LABEL_BUFFER ) buffer VisibleLabelBuffer
{
    int visibleLabel[];
};

layout (location = BRICK_SIZE ) uniform int brick_size;
layout (location = BRICK_MARGIN) uniform int brick_margin;
layout (location = BRICK_DIM ) uniform ivec3 brick_dim;
layout (location = MASK_DIM ) uniform ivec3 mask_dim;
layout (location = MASK_TEXTURE ) uniform sampler3D mask_sampler;
layout (location = BRICK_RANGE_MIN) uniform ivec3 brick_range_min;
layout (location = BRICK_RANGE_DIM) uniform ivec3 brick_range_dim;
layout (location = VISIBLE_LABEL_COUNT ) uniform int visble_label_count;

bool check_out_aabb(ivec3 pos , ivec3 dim);
bool check_outside(vec3 point, vec3 boundary);

void statistic_mask_info_in_cube(ivec3 begin , ivec3 end , in sampler3D mask_sampler , out int label_code)
{
    int label_max = 0;
    int label_min = 255;
    bool all_air = true;
    for(uint z = begin.z ; z < end.z ; ++z)
    {
        for(uint y = begin.y ; y < end.y ; ++y)
        {
            for(uint x = begin.x ; x<end.x ; ++x)
            {
                int label = int( texelFetch(mask_sampler, ivec3(x,y,z), 0).r*255.0);
                if(label == 0)
                {
                    continue;
                }
                for(int i = 0 ; i< visble_label_count ; ++i )
                {
                    if(label == visibleLabel[i])
                    {
                        all_air = false;
                        label_min = min(label_min , label);
                        label_max = max(label_max , label);
                        break;
                    }
                }
            }
        }
    }

    if(all_air)
    {
        label_code = 0;
    }
    else if(label_min == label_max)
    {
        label_code = label_min;
    }
    else
    {
        label_code = 255;
    }
}

void main()
{
    if(check_out_aabb(ivec3(gl_GlobalInvocationID) ,brick_range_dim ))
    {
        return;
    }

    ivec3 current_id = ivec3(gl_GlobalInvocationID.xyz) + brick_range_min;

    const uint real_global_invocation_id = current_id.z * brick_dim.x * brick_dim.y +
        current_id.y * brick_dim.x + current_id.x;

    ivec3 begin = ivec3(gl_GlobalInvocationID)*ivec3(brick_size);
    ivec3 end = begin + ivec3(brick_size);
    begin -= ivec3(brick_margin);
    end += ivec3(brick_margin);

    begin = max(begin ,ivec3(0));
    begin = min(begin ,mask_dim);

    end = max(end ,ivec3(0));
    end = min(end ,mask_dim);

    int label_code = 0;
    statistic_mask_info_in_cube(begin , end, mask_sampler , label_code);
    brickMaskInfo[real_global_invocation_id].labelCode = label_code;
}