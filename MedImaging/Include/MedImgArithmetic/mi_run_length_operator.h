#ifndef MED_IMAGING_RUN_LENGTH_OPERATOR_H
#define MED_IMAGING_RUN_LENGTH_OPERATOR_H

#include <vector>

MED_IMAGING_BEGIN_NAMESPACE

class RunLengthOperator
{
public:
    Arithmetic_Export static std::vector<unsigned int> encode(const std::vector<unsigned char>& to_be_encoded );
    Arithmetic_Export static std::vector<unsigned int> encode(const unsigned char* mask_array_pointer, const size_t total_number_of_voxels);
              
    Arithmetic_Export static std::vector<unsigned char> decode(const std::vector<unsigned int>& to_be_decoded );
};

MED_IMAGING_END_NAMESPACE
#endif