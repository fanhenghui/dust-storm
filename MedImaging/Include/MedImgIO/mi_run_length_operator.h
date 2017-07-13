#ifndef MED_IMAGING_RUN_LENGTH_OPERATOR_H
#define MED_IMAGING_RUN_LENGTH_OPERATOR_H

#include <vector>

MED_IMAGING_BEGIN_NAMESPACE
    class RunLengthOperator
    {
    public:
        IO_Export static std::vector<unsigned int> Encode(std::vector<unsigned int>& toBeEncoded );
        IO_Export static std::vector<unsigned int> Encode(const unsigned char* mask_array_pointer, const size_t total_number_of_voxels);
                  
        IO_Export static std::vector<unsigned char> Decode(std::vector<unsigned int>& toBeDecoded );

        //static void WriteLabels(std::string& file_name, std::vector<unsigned int>& labels);
        //static std::vector<unsigned char> ReadLabels(std::string& file_name);
    };
MED_IMAGING_END_NAMESPACE
#endif