#version 430

layout (local_size_x = 4 , local_size_y = 4) in;

layout (binding = IMAGE_ENTRY_POINT, rgba32f) uniform image2D image_entry_points;
layout (binding = IMAGE_EXIT_POINT, rgba32f) uniform image2D image_exit_points;

void main()
{
    const ivec2 img_coord = ivec2(gl_GlobalInvocationID.xy);
    if(img_coord.x > display_size.x -1  || img_coord.y > display_size.y -1)
    {
        return;
    }

}
