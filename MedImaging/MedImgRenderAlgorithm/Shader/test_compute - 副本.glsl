#version 430 core

layout (local_size_x = 4 , local_size_y = 4) in;

layout (binding = 0, rgba32f) uniform image2D imgEntryPoint;

//layout (location = 2) uniform uvec2 vDisplaySize;
//layout (location = 3) uniform vec3 vVolumeDim;

void main()
{
    const ivec2 vImgCoord = ivec2(gl_GlobalInvocationID.xy);

    imageStore(imgEntryPoint , vImgCoord , vec4(500,500,0, 0));


   // if(vImgCoord.x > vDisplaySize.x -1  || vImgCoord.y > vDisplaySize.y -1)
   // {
   //     return;
   // }

   //imageStore(imgEntryPoint , vImgCoord , vec4(500,500,0, 255));
}