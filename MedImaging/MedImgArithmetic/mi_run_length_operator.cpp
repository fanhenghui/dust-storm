#include "mi_run_length_operator.h"

MED_IMAGING_BEGIN_NAMESPACE

std::vector<unsigned int> RunLengthOperator::encode(const unsigned char* mask_array_pointer, const size_t total_number_of_voxels)
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

    return result; // let's rely on return-value-optimization
    // return std::move(result);
}

std::vector<unsigned int> RunLengthOperator::encode(const std::vector<unsigned char>& to_be_encoded)
{
    return RunLengthOperator::encode(to_be_encoded.data(), to_be_encoded.size());
}

std::vector<unsigned char> RunLengthOperator::decode(std::vector<unsigned int>& to_be_decoded)
{
    // count the voxels
    unsigned int total_number_of_voxels = 0;
    for (auto it = to_be_decoded.begin(); it != to_be_decoded.end(); it += 2)
    {
        total_number_of_voxels += (*it);
    }
    //std::cout << total_number_of_voxels << " labels are loaded\n";

    std::vector<unsigned char> result(total_number_of_voxels);

    // decode
    unsigned int current_index = 0;
    unsigned int voxel_bound = to_be_decoded[current_index];
    unsigned char current_label = static_cast<unsigned char>( to_be_decoded[current_index+1] );

    for (unsigned int voxel=0; voxel< total_number_of_voxels; ++voxel)
    {
        if (voxel == voxel_bound )
        {
            current_index += 2;
            voxel_bound += to_be_decoded[current_index];
            current_label = static_cast<unsigned char>( to_be_decoded[current_index+1] );
        }

        // populate the temporary information container
        result[voxel] = current_label;
    }

    return result; // std::move(result)
}

MED_IMAGING_END_NAMESPACE

    
