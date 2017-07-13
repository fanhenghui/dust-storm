#include "mi_run_length_operator.h"

MED_IMAGING_BEGIN_NAMESPACE

std::vector<unsigned int> RunLengthOperator::Encode(const unsigned char* mask_array_pointer, const size_t total_number_of_voxels)
{
    std::vector<unsigned int> result;
    if (mask_array_pointer != nullptr)
    {
        // encode with run-length in the format of <times, value>
        unsigned int cnt = 1; unsigned char val = mask_array_pointer[0];
        for (int voxel=1; voxel<total_number_of_voxels; ++voxel)
        {
            if (val == mask_array_pointer[voxel])
            {
                ++cnt;
            }
            else
            {
                result.push_back(cnt);
                result.push_back( static_cast<unsigned int>(val) );

                val = mask_array_pointer[voxel];
                cnt = 1;
            }
        }

        // record the last pair
        result.push_back(cnt);
        result.push_back(static_cast<unsigned int>(val));

        unsigned int sum_voxels = 0;
        for (auto it=result.begin(); it != result.end(); it += 2)
        {
            sum_voxels += (*it);
        }
        std::cout << sum_voxels << " voxels get counted\n";
    }

    return std::move(result);
}

std::vector<unsigned char> RunLengthOperator::Decode(std::vector<unsigned int>& toBeDecoded)
{
    std::vector<unsigned char> result;

    return std::move(result);
}
MED_IMAGING_END_NAMESPACE

    
