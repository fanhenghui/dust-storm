#version 430

#define BUFFER_BINDING_VISIBLE_LABEL_ARRAY 2
#define BUFFER_BINDING_MASK_OVERLAY_COLOR_BUCKET 3

layout (std430 , binding = BUFFER_BINDING_VISIBLE_LABEL_ARRAY) buffer VisibleLabelArray
{
    int visibleLabelArray[];
};
uniform int visible_label_count;

layout (std430 , binding = BUFFER_BINDING_MASK_OVERLAY_COLOR_BUCKET) buffer MaskOverlayColorBucket
{
    vec4 mask_overlay_color[];
};


void label_encode(int label , in out int mask_flag[4]);
bool label_decode(int label , int mask_flag[4]);


int global_active_label_code[4];

//Ray cast step code : 
//1 first sub data step 
//2 middle sub data step 
//4 last sub data step
vec4 mask_overlay(vec3 ray_start, vec3 ray_dir, float start_step, float end_step, vec4 integral_color,
        sampler3D volume_sampler,  sampler3D mask_sampler,   vec3 sub_data_dim , vec3 sub_data_offset , int ray_cast_step_code)
{
    vec4 current_integral_color = integral_color;

    if(0!= (ray_cast_step_code & 0x0001))//First sub data
    {
        global_active_label_code[0] = 0;
        global_active_label_code[1] = 0;
        global_active_label_code[2] = 0;
        global_active_label_code[3] = 0;

        current_integral_color.w = 0;
    }

    if(0!= (ray_cast_step_code & 0x0002))//Middle sub data
    {

    }

    vec3 sample_pos;
    int label = 0;
    for (float i = start_step ; i < end_step ; ++i)
    {
        sample_pos = ray_start + ray_dir * i;

        vec3 actual_sample_pos = (sample_pos + sub_data_offset + vec3(0.5,0.5,0.5))/sub_data_dim;
        label = int(texture(mask_sampler, actual_sample_pos).r*255);//Using default nearest interpolation
        if(label != 0)
        {
            //Encoding
            label_encode(label , global_active_label_code);
        }
    }

    //Last sub data decode label and blend output
    bool exist_active_label = false;
    if(0!= (ray_cast_step_code & 0x0004))
    {
        vec4 label_color = vec4(0,0,0,0);
        for(int i = 0 ; i < visible_label_count ; ++i)
        {
            int cur_label = visibleLabelArray[i];
        
            //Decoding
            bool hitted = label_decode(cur_label , global_active_label_code);

            //Blending
            if(hitted)
            {
                exist_active_label = true;
                label_color = mask_overlay_color[cur_label];
                current_integral_color.xyz += (1 - current_integral_color.w) * label_color.w * label_color.xyz;
                current_integral_color.w += (1 - current_integral_color.w) * label_color.w;
            }
        }

        current_integral_color.w = 1.0;
    }

    return current_integral_color;
}
