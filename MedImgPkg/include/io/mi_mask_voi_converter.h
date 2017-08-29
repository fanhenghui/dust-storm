#ifndef MEDIMGIO_MI_MASK_VOI_CONVERTER_H
#define MEDIMGIO_MI_MASK_VOI_CONVERTER_H

#include "io/mi_io_export.h"
#include "io/mi_voi.h"

#include <vector>

MED_IMG_BEGIN_NAMESPACE

class MaskVOIConverter
{
public:
    //Diameter physical distance
    //Center volume coordinate
    IO_Export static std::vector<VOISphere> convert_label_2_sphere(const std::vector<unsigned char>& labels, const unsigned int dim[3], const double spacing[3], const double origin[3]);
};
MED_IMG_END_NAMESPACE

#endif