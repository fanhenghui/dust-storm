#ifndef MEDIMGARITHMETIC_RUN_LENGTH_OPERATOR_H
#define MEDIMGARITHMETIC_RUN_LENGTH_OPERATOR_H

#include "arithmetic/mi_arithmetic_export.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export RunLengthOperator {
public:
    static std::vector<unsigned int> encode(const std::vector<unsigned char>& to_be_encoded);
    static std::vector<unsigned int> encode(const unsigned char* mask_array_pointer,
            const size_t total_number_of_voxels);

    static std::vector<unsigned char> decode(const std::vector<unsigned int>& to_be_decoded);
    static int decode(unsigned int* code_buffer , unsigned int code_buffer_len , 
        unsigned char* target_mask , unsigned int mask_buffer_len);
};

MED_IMG_END_NAMESPACE
#endif