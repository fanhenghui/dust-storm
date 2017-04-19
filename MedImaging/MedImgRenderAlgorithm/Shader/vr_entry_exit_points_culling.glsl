#version 430

layout (local_size_x = 4 , local_size_y = 4) in;

layout (binding = IMAGE_ENTRY_POINT, rgba32f) uniform image2D imgEntryPoint;
layout (binding = IMAGE_EXIT_POINT, rgba32f) uniform image2D imgExitPoint;

void main()
{
    const ivec2 vImgCoord = ivec2(gl_GlobalInvocationID.xy);
    if(vImgCoord.x > vDisplaySize.x -1  || vImgCoord.y > vDisplaySize.y -1)
    {
        return;
    }

    vec4 vEntryPoint = imageLoad(imgEntryPoint , vImgCoord).xyzw;
    vec4 vExitPoint = imageLoad(imgExitPoint , vImgCoord).xyzw;

}
