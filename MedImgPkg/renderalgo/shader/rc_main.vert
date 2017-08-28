#version 430

layout (location = 0) in vec4 vertex;
//layout (location = 1) in vec4 color;

void main()
{
    gl_Position = vec4(vertex.xy,0.0,1.0);
}