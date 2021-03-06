#version 430

layout (location = 0) out vec4 oFragColor;
layout (location = 1) out vec4 oFragPara;

#define IMG_BINDING_ENTRY_POINTS  0
#define IMG_BINDING_EXIT_POINTS  1

layout (binding = IMG_BINDING_ENTRY_POINTS, rgba32f) readonly uniform image2D image_entry_points;
layout (binding = IMG_BINDING_EXIT_POINTS, rgba32f) readonly uniform image2D image_exit_points;

uniform vec3 volume_dim;
uniform sampler3D volume_sampler;
uniform sampler3D mask_sampler;
uniform float sample_step;
uniform int quarter_canvas;
uniform vec3 eye_position;
uniform int ray_align_to_view_plane;
uniform int jittering;
uniform sampler2D random_sampler;

void preprocess(out vec3 ray_start,out vec3 ray_dir_sample_step, out float start_step, out float end_step)
{
    ivec2 frag_coord = ivec2(0,0);
    if(1 == quarter_canvas) {
        frag_coord = ivec2(gl_FragCoord.x*2, gl_FragCoord.y*2);
    } else {
        frag_coord = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    }
     
    vec3 start_point = imageLoad(image_entry_points, frag_coord.xy).xyz;
    vec3 end_point = imageLoad(image_exit_points, frag_coord.xy).xyz;

    vec3 ray_dir = end_point - start_point;
    vec3 ray_dir_norm = normalize(ray_dir);
    float ray_length = length(ray_dir);

    if(ray_length < 1e-5)
    {
        discard;
    }

    float adjust = 0.0f;
    if (1 == ray_align_to_view_plane) {
        vec3 v0 = start_point - eye_position;
        float len = dot(v0, ray_dir_norm);
        adjust = len/sample_step;
        adjust = (ceil(adjust) - adjust)*sample_step;
        ray_length = ray_length - adjust;

        if(ray_length < 1e-5)
        {
            discard;
        }    
    }
    
    ray_start = start_point + adjust*ray_dir_norm;

    if (1 == jittering) {
        vec2 img_size = textureSize(random_sampler,0);
        vec2 frag_coord_norm = vec2(frag_coord.xy) / img_size;
        ray_start += texture(random_sampler, frag_coord_norm).x * ray_dir_norm;
    }
    
    ray_dir_sample_step = ray_dir_norm * sample_step;
    start_step = 0;
    end_step = ray_length/ sample_step;
}

//Ray cast step code : 
//1 first sub data step 
//2 middle sub data step 
//4 last sub data step
vec4 ray_cast(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
    sampler3D volume_sampler, sampler3D mask_sampler, vec3 sub_data_dim, vec3 sub_data_offset, vec3 sample_shift, 
    int ray_cast_step_code, out vec3 ray_end);

vec4 mask_overlay(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
        sampler3D volume_sampler, sampler3D mask_sampler, vec3 sub_data_dim, vec3 sub_data_offset, int ray_cast_step_code);

void main()
{
    vec3 ray_start = vec3(0,0,0);
    vec3 ray_dir_sample_step = vec3(1,0,0);
    float end_step = 0;
    float start_step = 0;
    vec3 ray_end = vec3(0,0,0);
    vec4 integral_color = vec4(0,0,0,0);

    preprocess(ray_start, ray_dir_sample_step, start_step, end_step);

    integral_color = ray_cast(
        ray_start, 
        ray_dir_sample_step, 
        start_step, 
        end_step, 
        integral_color , 
        volume_sampler , 
        mask_sampler , 
        volume_dim,
        vec3(0.0), 
        vec3(1.0)/volume_dim,
        5,
        ray_end);

    oFragColor = mask_overlay(
        ray_start, 
        ray_dir_sample_step, 
        start_step, 
        end_step, 
        integral_color , 
        volume_sampler , 
        mask_sampler , 
        volume_dim,
        vec3(0.0),
        5);

    oFragPara = vec4(ray_end/volume_dim,0.0);
}