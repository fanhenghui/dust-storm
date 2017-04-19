#version 430 core

layout (local_size_x = 4 , local_size_y = 4) in;

layout (binding = 0, rgba8) uniform image2D imgEntryPoint;

void main()
{
    const ivec2 vImgCoord = ivec2(gl_GlobalInvocationID.xy);

    imageStore(imgEntryPoint , vImgCoord , vec4(1,1,1, 1));
}