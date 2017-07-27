#version 430 core

layout (location = 0) in vec4 vertex ;
layout (location = 1) in vec4 color;
layout (location = 0) uniform mat4 mat_mvp;
out vec4 out_color;

void main()
{
    out_color = vertex;
    gl_Position = mat_mvp*vertex;
}