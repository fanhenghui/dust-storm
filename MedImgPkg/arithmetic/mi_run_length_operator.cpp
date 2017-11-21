#include "mi_run_length_operator.h"
#include "mi_arithmetic_logger.h"
#include <fstream>
#include "mi_arithmetic_logger.h"

MED_IMG_BEGIN_NAMESPACE

std::vector<unsigned int> RunLengthOperator::encode(const unsigned char* mask_array_pointer,
        const unsigned int total_number_of_voxels) {
    std::vector<unsigned int> result;

    if (mask_array_pointer != nullptr) {
        // encode with run-length in the format of <times, value>
        unsigned int cnt = 1;
        unsigned char val = mask_array_pointer[0];

        for (unsigned int voxel = 1; voxel < total_number_of_voxels; ++voxel) {
            if (val == mask_array_pointer[voxel]) {
                ++cnt;
            } else {
                result.push_back(cnt);
                result.push_back(static_cast<unsigned int>(val));

                val = mask_array_pointer[voxel];
                cnt = 1;
            }
        }

        // record the last pair
        result.push_back(cnt);
        result.push_back(static_cast<unsigned int>(val));

        unsigned int sum_voxels = 0;

        for (auto it = result.begin(); it != result.end(); it += 2) {
            sum_voxels += (*it);
        }

        MI_ARITHMETIC_LOG(MI_DEBUG) << sum_voxels << " voxels get counted.";
    }

    return result; // let's rely on return-value-optimization
    // return std::move(result);
}

std::vector<unsigned int> RunLengthOperator::encode(const std::vector<unsigned char>&
        to_be_encoded) {
    return RunLengthOperator::encode(to_be_encoded.data(), to_be_encoded.size());
}

std::vector<unsigned char> RunLengthOperator::decode(const std::vector<unsigned int>&
        to_be_decoded) {
    // count the voxels
    unsigned int total_number_of_voxels = 0;

    for (auto it = to_be_decoded.begin(); it != to_be_decoded.end(); it += 2) {
        total_number_of_voxels += (*it);
    }

    std::vector<unsigned char> result(total_number_of_voxels);

    // decode
    unsigned int current_index = 0;
    unsigned int voxel_bound = to_be_decoded[current_index];
    unsigned char current_label = static_cast<unsigned char>(to_be_decoded[current_index + 1]);

    for (unsigned int voxel = 0; voxel < total_number_of_voxels; ++voxel) {
        if (voxel == voxel_bound) {
            current_index += 2;
            voxel_bound += to_be_decoded[current_index];
            current_label = static_cast<unsigned char>(to_be_decoded[current_index + 1]);
        }

        // populate the temporary information container
        result[voxel] = current_label;
    }

    return result; // std::move(result)
}

int RunLengthOperator::decode(
    unsigned int* code_buffer , 
    unsigned int code_buffer_len , 
    unsigned char* target_mask , 
    unsigned int mask_buffer_len) {
    unsigned int voxel_num = 0;
    for (unsigned int i = 0; i < code_buffer_len; i += 2) {
        voxel_num += code_buffer[i];
    }

    if (voxel_num != mask_buffer_len) {
        MI_ARITHMETIC_LOG(MI_ERROR) << "decode rle file failed beause input buffer size is not match.";
        return -1;
    }

    unsigned int cur_idx = 0;
    unsigned int voxel_bound = code_buffer[cur_idx];
    unsigned char cur_label = static_cast<unsigned char>(code_buffer[cur_idx + 1]);

    for (unsigned int voxel = 0; voxel < voxel_num; ++voxel) {
        if (voxel == voxel_bound) {
            cur_idx += 2;
            voxel_bound += code_buffer[cur_idx];
            cur_label = static_cast<unsigned char>(code_buffer[cur_idx + 1]);
        }
        // populate the temporary information container
        target_mask[voxel] = cur_label;
    }
    return 0;
}

int RunLengthOperator::decode(const std::string& rle_file, unsigned char* target_mask, unsigned int target_mask_size) {
    std::ifstream in(rle_file);
    if (!in.is_open()) {
        MI_ARITHMETIC_LOG(MI_ERROR) << "open rle file: " << rle_file << " failed.";
        return -1;
    }

    // get size in bytes
    in.seekg (0, in.end);
    const unsigned int file_size = in.tellg();
    in.seekg (0, in.beg);

    // prepare the buffer and copy into it
    const int number_of_entries = file_size/sizeof(int);
    std::vector<unsigned int> labels(number_of_entries);
    char* buffer = reinterpret_cast<char*>(labels.data());
    in.read(buffer, file_size);
    in.close();

    return RunLengthOperator::decode((unsigned int*)buffer, file_size/sizeof(unsigned int), target_mask, target_mask_size);
}

MED_IMG_END_NAMESPACE

    

