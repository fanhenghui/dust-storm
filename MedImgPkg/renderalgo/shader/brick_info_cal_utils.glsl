#version 430

bool check_out_aabb(ivec3 pos , ivec3 dim)
{
    bvec3 compare_max = greaterThanEqual (pos, dim);
    return any(compare_max);
}

bool check_outside(vec3 point, vec3 boundary)
{
    bvec3 compare_min = lessThanEqual(point, vec3(0.0));
    bvec3 compare_max = greaterThanEqual (point, boundary);
    return any(compare_min) || any(compare_max);
}

void statistic_volume_info_in_cube(ivec3 begin , ivec3 end ,  sampler3D volume_sampler , float volume_reg_param , out float min0 , out float max0)
{
    min0 = volume_reg_param;
    max0 = -volume_reg_param;
    for(uint z = begin.z ; z < end.z ; ++z)
    {
        for(uint y = begin.y ; y < end.y ; ++y)
        {
            for(uint x = begin.x ; x<end.x ; ++x)
            {
                float v = texelFetch(volume_sampler, ivec3(x,y,z), 0).r*volume_reg_param;
                min0 = min(min0 , v);
                max0 = max(max0 , v);
            }
        }
    }
}

