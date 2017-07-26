#ifndef MED_IMAGING_ARITHMETIC_MASK_TO_SPHERE_H_
#define MED_IMAGING_ARITHMETIC_MASK_TO_SPHERE_H_

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgIO/mi_voi.h"

#include <vector>

MED_IMAGING_BEGIN_NAMESPACE

class MaskVOIConverter
{
public:
    //Diameter physical distance
    //Center volume coordinate
    IO_Export static std::vector<VOISphere> convert_label_2_sphere(const std::vector<unsigned char>& labels, const unsigned int dim[3], const double spacing[3], const double origin[3]);
};
MED_IMAGING_END_NAMESPACE

#endif