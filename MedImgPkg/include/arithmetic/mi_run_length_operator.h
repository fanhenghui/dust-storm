#ifndef MEDIMGARITHMETIC_RUN_LENGTH_OPERATOR_H
#define MEDIMGARITHMETIC_RUN_LENGTH_OPERATOR_H

#include "arithmetic/mi_arithmetic_export.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class RunLengthOperator {
public:
    Arithmetic_Export static std::vector<unsigned int> encode(const std::vector<unsigned char>&
            to_be_encoded);
    Arithmetic_Export static std::vector<unsigned int> encode(const unsigned char* mask_array_pointer,
            const size_t total_number_of_voxels);

    Arithmetic_Export static std::vector<unsigned char> decode(const std::vector<unsigned int>&
            to_be_decoded);
};

MED_IMG_END_NAMESPACE
#endif