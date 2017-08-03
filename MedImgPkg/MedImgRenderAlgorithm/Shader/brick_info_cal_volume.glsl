#version 430 

#define BRICK_VOLUME_INFO_BUFFER 0
#define BRICK_SIZE 1
#define BRICK_MARGIN 2
#define BRICK_DIM 3
#define VOLUME_DIM 4
#define VOLUME_TEXTURE 5
#define VOLUME_MIN_SCALAR 6
#define VOLUME_REGULATE_PARAMETER 7

//One local work group handle 5x5x1 25 bricks
layout (local_size_x = 5 , local_size_y = 5 , local_size_z = 1) in;

struct BrickVolumeInfo
{
    float min;
    float max;
};
layout (std430 , binding = BRICK_VOLUME_INFO_BUFFER ) buffer BrickVolumeInfoInOut
{
    BrickVolumeInfo brickVolumeInfo[];
};

layout (location = BRICK_SIZE ) uniform int brick_size;
layout (location = BRICK_MARGIN) uniform int brick_margin;
layout (location = BRICK_DIM ) uniform ivec3 brick_dim;
layout (location = VOLUME_DIM ) uniform ivec3 volume_dim;
layout (location = VOLUME_TEXTURE ) uniform sampler3D volume_sampler;
layout (location = VOLUME_MIN_SCALAR) uniform float volume_min_scalar;
layout (location = VOLUME_REGULATE_PARAMETER) uniform float volume_regulate_param;

bool check_out_aabb(ivec3 pos , ivec3 dim);
bool check_outside(vec3 point, vec3 boundary);
void statistic_volume_info_in_cube(ivec3 beign , ivec3 end, sampler3D volume_sampler , float volume_regulate_param , out float min , out float max);

void main()
{
    if(check_out_aabb(ivec3(gl_GlobalInvocationID) ,brick_dim ))
    {
        return;
    }

    const uint real_global_invocation_id = gl_GlobalInvocationID.z * brick_dim.x * brick_dim.y +
        gl_GlobalInvocationID.y * brick_dim.x  + gl_GlobalInvocationID.x;

    ivec3 begin = ivec3(gl_GlobalInvocationID)*ivec3(brick_size);
    ivec3 end = begin + ivec3(brick_size);
    begin -= ivec3(brick_margin);
    end += ivec3(brick_margin);

    begin = max(begin ,ivec3(0));
    begin = min(begin ,volume_dim);

    end = max(end ,ivec3(0));
    end = min(end ,volume_dim);

    float min0 = 0;
    float max0 = 0;
    statistic_volume_info_in_cube(begin , end, volume_sampler , volume_regulate_param , min0 , max0);

    brickVolumeInfo[real_global_invocation_id].min = min0+ volume_min_scalar;
    brickVolumeInfo[real_global_invocation_id].max = max0 + volume_min_scalar;
}