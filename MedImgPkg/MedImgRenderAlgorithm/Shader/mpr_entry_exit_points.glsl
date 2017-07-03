#version 430

#define IMAGE_ENTRY_POINT 0
#define IMAGE_EXIT_POINT 1
#define DISPLAY_SIZE 2
#define VOLUME_DIM 3
#define MVP_INVERSE 4
#define THICKNESS 5
#define RAY_DIRECTION 6

layout (local_size_x = 4 , local_size_y = 4) in;

layout (binding = IMAGE_ENTRY_POINT, rgba32f) uniform image2D image_entry_points;
layout (binding = IMAGE_EXIT_POINT, rgba32f) uniform image2D image_exit_points;

layout (location = DISPLAY_SIZE) uniform uvec2 display_size;
layout (location = VOLUME_DIM) uniform vec3 volume_dim;
layout (location = MVP_INVERSE) uniform mat4 mat_mvp_inv;
layout (location = THICKNESS) uniform float thickness;
layout (location = RAY_DIRECTION) uniform vec3 ray_dir;

/// Get exaclty point
float ray_intersect_brick(vec3 initialPt, vec3 brick_min, vec3 brick_dim, vec3 ray_dir, 
    out float start_step, out float end_step)
{
    vec3 invR = 1.0 / (ray_dir); 

    vec3 bottom =  (brick_min - initialPt);
    vec3 top =  (brick_min + brick_dim - initialPt);
    vec3 tbot = invR * bottom;
    vec3 ttop = invR * top; 

    vec3 tmin = min(tbot, ttop);
    vec3 tmax = max(tbot, ttop);
    float tnear = max(max(tmin.x,tmin.y),tmin.z);
    float tfar = min(min(tmax.x,tmax.y),tmax.z);

    start_step = tnear;   
    start_step = max(start_step, 0.0);
    end_step = tfar;

    return tnear - start_step;
}

bool check_outside(vec3 point, vec3 boundary)
{
    bvec3 compare_min = lessThan(point, vec3(0.0, 0.0, 0.0));
    bvec3 compare_max = greaterThan(point, boundary);
    return any(compare_min) || any(compare_max);
}

void main()
{
    const ivec2 img_coord = ivec2(gl_GlobalInvocationID.xy);
    if(img_coord.x > display_size.x -1  || img_coord.y > display_size.y -1)
    {
        return;
    }

    //imageStore(image_entry_points , img_coord , vec4(img_coord.x,img_coord.y,100, 255));
    //return;

    float x = (float(img_coord.x) +0.5)/float(display_size.x);
    float y = (float(img_coord.y) +0.5)/float(display_size.y);

    vec3 pos_ndc = vec3(x*2.0-1.0 , y*2.0-1.0 , 0.0);//not DC to NDC , just NDC to memory

    vec4 central4 = mat_mvp_inv * vec4(pos_ndc,1.0);
    vec3 central = central4.xyz / central4.w;

    vec3 entry_point, exit_point;
    if(thickness <= 1.0)
    {
        entry_point = central ;
        exit_point  = central + ray_dir * thickness*0.5;
    }
    else
    {
        entry_point = central - ray_dir * thickness *0.5 ;
        exit_point  = central + ray_dir * thickness * 0.5 ;
    }
    

    //imageStore(image_entry_points , img_coord , vec4(entry_point, 1.0f));
    //imageStore(image_exit_points , img_coord , vec4(exit_point, 1.0f));
    //return;

    float entry_step = 0.0;
    float exit_step = 0.0;

    vec3 entry_intersection = entry_point;
    vec3 exit_intersection = exit_point;

    ray_intersect_brick(entry_point, vec3(0,0,0),volume_dim, ray_dir, entry_step, exit_step);

    //Entry point outside
    if( check_outside(entry_point, volume_dim) )
    {
        if(entry_step >= exit_step || entry_step < 0 || entry_step > thickness)// check entry points in range of thickness and volume
        {
            exit_step = -1.0;
            imageStore(image_entry_points , img_coord , vec4(0,0,0, -1.0f));
            imageStore(image_exit_points , img_coord , vec4(0,0,0, -1.0f));
            return;
        }
        entry_intersection = entry_point + entry_step * ray_dir;
    }

    //Exit point outside
    if( check_outside(exit_point, volume_dim) )
    {
        if(entry_step >= exit_step)
        {
            exit_step = -1.0;
            imageStore(image_entry_points , img_coord , vec4(0,0,0, -1.0f));
            imageStore(image_exit_points , img_coord , vec4(0,0,0, -1.0f));
            return;
        }
        exit_intersection= entry_point + exit_step * ray_dir;
    }

    imageStore(image_entry_points , img_coord , vec4(entry_intersection, 1.0f));
    imageStore(image_exit_points , img_coord , vec4(exit_intersection, 1.0f));

    //imageStore(image_entry_points , img_coord , vec4(255,0,0, 1.0f));
    //imageStore(image_exit_points , img_coord , vec4(exit_intersection, 1.0f));

}