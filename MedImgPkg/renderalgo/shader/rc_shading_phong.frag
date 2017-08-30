#version 430

#define BUFFER_BINDING_MATERIAL_BUCKET 5

uniform mat4 mat_normal;
uniform vec3 spacing;

//single point light
uniform vec3 light_position;
uniform vec4 ambient_color;

struct Material
{
    vec4 diffuse_color;
    vec4 specular_color;
    float shininess;
    float _reserve0;
    float _reserve1;
    float _reserve2;
};

layout(std430 , binding = BUFFER_BINDING_MATERIAL_BUCKET) buffer MaterialBucket
{
    Material material[];
};

vec3 calculate_gradient(sampler3D sampler, vec3 sample_pos, vec3 sample_shift, vec3 spacing)
{
    vec4 shift = vec4(sample_shift / spacing,0.0);

    vec3 gradient = vec3(
        texture(sampler, sample_pos + shift.xww).r - texture(sampler, sample_pos - shift.xww).r,
        texture(sampler, sample_pos + shift.wyw).r - texture(sampler, sample_pos - shift.wyw).r,
        texture(sampler, sample_pos + shift.wwz).r - texture(sampler, sample_pos - shift.wwz).r);

    return gradient;
}

vec4 shade(vec3 sample_pos, vec4 input_color, vec3 ray_dir , sampler3D sampler , vec3 pos_in_volume , vec3 sample_shift , int label)
{
    vec3 normal = calculate_gradient(sampler,sample_pos ,sample_shift, spacing); 
    normal = (mat_normal * vec4(normal,0.0)).xyz;
    normal = normalize(normal);

    vec3 view_dir = (mat_normal * vec4(-ray_dir,0.0)).xyz;
    view_dir = normalize(view_dir);

    vec3 light_dir = light_position - pos_in_volume;
    light_dir = normalize(light_dir);
    light_dir = (mat_normal * vec4(light_dir,0.0)).xyz;
    light_dir = normalize(light_dir);

    vec3 ambient_part = ambient_color.xyz*ambient_color.w*input_color.xyz;
    float ln = dot(light_dir, normal);
    if (ln < 0.0) 
    {
        normal = -normal;
        ln = -ln;
    }

    float diffuse = max(ln, 0.0);
    vec3 diffuse_part = diffuse * material[label].diffuse_color.xyz * material[label].diffuse_color.w *input_color.xyz;

    //Classic phong
    vec3 r = reflect(-light_dir, normal);
    r = normalize(r);
    float specular = max(dot(r, view_dir), 0.0);

    //Blinn-Phong
    /*vec3 h= view_dir + light_dir;
    h = normalize(h);
    float specular = max(dot(h, normal), 0.0);*/

    specular = pow(specular, material[label].shininess);
    vec3 specular_part = specular * material[label].specular_color.xyz * material[label].specular_color.w *input_color.xyz;

    vec3 output_color = ambient_part + diffuse_part + specular_part;

    //silhouettes enhance alpha
    float fn = 1.0 - ln;
    float kss = 1;
    float kse = 0.5;
    float ksc = 0.0;
    float alpha =  input_color.w*(0.5 + kss * pow(fn, kse));

    alpha = clamp(alpha , 0, 1);
    output_color = clamp(output_color , 0 , 1);

    return vec4(output_color , alpha);
}
