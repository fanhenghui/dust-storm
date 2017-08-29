#include "io/mi_mask_voi_converter.h"

#include <algorithm>

#include "io/mi_voi.h"
#include "arithmetic/mi_aabb.h"

MED_IMG_BEGIN_NAMESPACE

static void get_structured_coordinates(const unsigned int idx, const unsigned int dim[3], unsigned int ijk[3] )
{
    unsigned int N12 = dim[0]*dim[1];
    ijk[2] = idx/N12;
    ijk[1] = (idx-ijk[2]*N12)/dim[1];
    ijk[0] = idx-ijk[2]*N12-ijk[1]*dim[1];
}

std::vector<VOISphere> MaskVOIConverter::convert_label_2_sphere(const std::vector<unsigned char>& labels, const unsigned int dim[3], const double spacing[3], const double origin[3])
{
    AABBUI* aabb_bucket[255];
    for (int i = 0 ; i< 255 ; ++i)  
    {
        aabb_bucket[i] = nullptr;
    }

    for (unsigned int voxel=0; voxel<labels.size(); ++voxel)
    {
        unsigned char current_label = labels[voxel];
        if (current_label == 0)
        {
            continue;
        }
        else
        {
            unsigned int ijk[3] = {0,0,0};
            get_structured_coordinates(voxel, dim, ijk);

            if (aabb_bucket[current_label]) // we have this label
            {
                //update aabbs at idx
                AABBUI* aabb = aabb_bucket[current_label];
                for (int i=0; i<3; ++i)
                {
                    if (aabb->_min[i] > ijk[i])
                    {
                        aabb->_min[i] = ijk[i];
                    }

                    if (aabb->_max[i] < ijk[i])
                    {
                        aabb->_max[i] = ijk[i];
                    }
                }
            }
            else
            {
                AABBUI* new_aabb = new AABBUI;
                aabb_bucket[current_label] = new_aabb;
                for (int i=0; i<3; ++i)
                {
                    new_aabb->_min[i] = ijk[i];
                    new_aabb->_max[i] = ijk[i];
                }
            }
        }
    }

    int label_num = 0;
    VOISphere* sphere_bucket[255];
    for (int i = 0 ; i< 255; ++i)
    {
        if (aabb_bucket[i])
        {
            sphere_bucket[i] = new VOISphere;
            sphere_bucket[i]->center.x = (aabb_bucket[i]->_min[0] + aabb_bucket[i]->_max[0])*0.5;
            sphere_bucket[i]->center.y = (aabb_bucket[i]->_min[1] + aabb_bucket[i]->_max[1])*0.5;
            sphere_bucket[i]->center.z = (aabb_bucket[i]->_min[2] + aabb_bucket[i]->_max[2])*0.5;

            sphere_bucket[i]->diameter = 0.0f;
            ++label_num;
        }
        else
        {
            sphere_bucket[i] = nullptr;
        }
    }

    for (unsigned int voxel=0; voxel<labels.size(); ++voxel)
    {
        unsigned char current_label = labels[voxel];
        if (current_label == 0)
        {
            continue;
        }
        else
        {
            unsigned int ijk[3] = {0,0,0};
            get_structured_coordinates(voxel, dim, ijk);

            if (sphere_bucket[current_label]) // we have this label
            {
                VOISphere* sphere = sphere_bucket[current_label];
                double dx = (ijk[0] - sphere->center.x)*spacing[0];
                double dy = (ijk[1] - sphere->center.y)*spacing[1];
                double dz = (ijk[2] - sphere->center.z)*spacing[2];
                double distance = sqrt(dx*dx + dy*dy + dz*dz);
                if ( distance > sphere->diameter*0.5f)
                {
                    sphere->diameter = distance*2.0f;
                }
            }
        }
    }

    std::vector<VOISphere> ret(label_num);
    int cur_label_num =0;
    for (int i = 0 ;i < 255 ; ++i)
    {
        if (sphere_bucket[i])
        {
            ret[cur_label_num] = VOISphere(*sphere_bucket[i]);
            /*ret[cur_label_num].center.x = ret[cur_label_num].center.x*spacing[0] + origin[0];
            ret[cur_label_num].center.y = ret[cur_label_num].center.y*spacing[1] + origin[1];
            ret[cur_label_num].center.z = ret[cur_label_num].center.z*spacing[2] + origin[2];*/
            ++cur_label_num;
        }
        if (cur_label_num > label_num)
        {
            break;
        }

    }

    for (int i = 0; i< 255 ; ++i)
    {
        if (aabb_bucket[i])
        {
            delete aabb_bucket[i];
            aabb_bucket[i] = nullptr;
        }
        if (sphere_bucket[i])
        {
            delete sphere_bucket[i];
            sphere_bucket[i] = nullptr;
        }
    }

    return ret;
}

MED_IMG_END_NAMESPACE