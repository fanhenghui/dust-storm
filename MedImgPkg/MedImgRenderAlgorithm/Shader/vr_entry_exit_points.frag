#version 430 core

in vec4 in_color;
out vec4 out_color;

void main()
{
    out_color = in_color;
    out_color.a = 0.0;
}