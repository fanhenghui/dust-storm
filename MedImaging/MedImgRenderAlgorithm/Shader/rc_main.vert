#version 430

layout (location = 0) in vec4 vVertex;
//layout (location = 1) in vec4 vColor;

void main()
{
    gl_Position = vec4(vVertex.xy,0.0,1.0);
}