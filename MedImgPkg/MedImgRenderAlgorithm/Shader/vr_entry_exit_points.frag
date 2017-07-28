#version 430 core

in vec4 v_out_color;
out vec4 f_out_color;

void main()
{
    f_out_color = v_out_color;
    f_out_color.a = 0.0;
}