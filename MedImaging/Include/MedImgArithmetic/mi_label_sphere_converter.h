#ifndef MED_IMAGING_ARITHMETIC_LABEL_2_SPHERE_H_
#define MED_IMAGING_ARITHMETIC_LABEL_2_SPHERE_H_

#include <vector>

MED_IMAGING_BEGIN_NAMESPACE
struct VOISphere;

class Label2SphereConverter
{
public:
    Arithmetic_Export static std::vector<VOISphere> convert_label_2_sphere(const std::vector<unsigned char>& labels, const unsigned int dim[3], const double spacing[3], const double origin[3]);
};
MED_IMAGING_END_NAMESPACE
#endif