#version 430

layout (location = 0) out vec4 oFragColor;

#define IMG_BINDING_ENTRY_POINTS  0
#define IMG_BINDING_EXIT_POINTS  1

layout (binding = IMG_BINDING_ENTRY_POINTS, rgba32f) readonly uniform image2D image_entry_points;
layout (binding = IMG_BINDING_EXIT_POINTS, rgba32f) readonly uniform image2D image_exit_points;

uniform vec3 volume_dim;
uniform int test_code;


void main()
{
    ivec2 frag_coord = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    if(1 == test_code)
    {
        oFragColor = vec4(imageLoad(image_entry_points, frag_coord.xy).xyz/volume_dim , 1.0);
    }
    else
    {
        oFragColor = vec4(imageLoad(image_exit_points, frag_coord.xy).xyz/volume_dim , 1.0);
    }
}