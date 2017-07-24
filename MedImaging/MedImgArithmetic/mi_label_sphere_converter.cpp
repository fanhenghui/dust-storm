#include "MedImgArithmetic/mi_label_sphere_converter.h"

#include "MedImgIO/mi_voi.h"
#include "MedImgArithmetic/mi_aabb.h"

#include <algorithm>

MED_IMAGING_BEGIN_NAMESPACE

static void GetStructuredCoordinates(const unsigned int idx, const unsigned int dim[3], unsigned int ijk[3] )
{
    unsigned int N12 = dim[0]*dim[1];
    ijk[2] = idx/N12;
    ijk[1] = (idx-ijk[2]*N12)/dim[1];
    ijk[0] = idx-ijk[2]*N12-ijk[1]*dim[1];
}
std::vector<VOISphere> Label2SphereConverter::convert_label_2_sphere(const std::vector<unsigned char>& labels, const unsigned int dim[3], const double spacing[3], const double origin[3])
{
    std::vector<unsigned char> unique_labels;
    std::vector<AABBUI> aabbs;
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
            GetStructuredCoordinates(voxel, dim, ijk);

            unsigned int idx = std::distance(unique_labels.begin(), find(unique_labels.begin(), unique_labels.end(), current_label));
            if (idx < unique_labels.size()) // we have this label
            {
                //update aabbs at idx
                AABBUI& aabb = aabbs.at(idx);
                for (int i=0; i<3; ++i)
                {
                    if (aabb._min[i] > ijk[i])
                    {
                        aabb._min[i] = ijk[i];
                    }

                    if (aabb._max[i] < ijk[i])
                    {
                        aabb._max[i] = ijk[i];
                    }
                }
            }
            else
            {
                unique_labels.push_back(current_label);
                AABBUI new_aabb;
                for (int i=0; i<3; ++i)
                {
                    new_aabb._min[i] = ijk[i];
                    new_aabb._max[i] = ijk[i];
                }
                aabbs.push_back(new_aabb);
            }
        }
    }

    std::vector<VOISphere> ret;
    for (int sphere_idx = 0; sphere_idx<unique_labels.size(); ++sphere_idx)
    {
        double center[3] = {0.0, 0.0, 0.0};
        double radius = 0.0;
        AABBUI& aabb = aabbs.at(sphere_idx);
        for (int i=0; i<3; ++i)
        {
            radius = std::max(0.5 * spacing[i] * (aabb._max[i]-aabb._min[i]+1), radius);
            center[i] = origin[i] +  spacing[i] * 0.5 * (aabb._max[i]+aabb._min[i]);
        }

        // construct the sphere
        ret.push_back(
            VOISphere(Point3(center[0], center[1], center[2]), 2.0*radius));
    }
    return ret;
}
MED_IMAGING_END_NAMESPACE