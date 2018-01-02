#include <cassert>
#include "mi_md5.h"
#include "mbedtls/md5.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

int digest(const std::string& data, unsigned char(&md5_value)[16]) {
    if (data.empty()) {
        MI_IO_LOG(MI_ERROR) << "can't calculate empty string's md5.";
        return -1;
    }
    mbedtls_md5((unsigned char*)(data.c_str()), data.size(), md5_value);
    return 0;
}

int MD5::digest(const std::string& data, char(&hex_str)[32]) {
    if (data.empty()) {
        MI_IO_LOG(MI_ERROR) << "can't calculate empty string's md5.";
        return -1;
    }
    unsigned char byte_array[16] = { 0 };
    mbedtls_md5((unsigned char*)(data.c_str()), data.length(), byte_array);
    int j = 0;
    int tmp = 0;
    for (int i = 0; i < 16; ++i) {
        tmp = byte_array[i];
        hex_str[j++] = (tmp >> 4) & 0x0f; 
        hex_str[j++] = tmp & 0x0f;
    }
    assert(j == 32);

    for (int i = 0; i < 32; ++i) { 
        if (hex_str[i] < 10) {
            hex_str[i] = '0' + hex_str[i];
        } else {
            hex_str[i] = 'a' + (hex_str[i]-10);
        }
    }
    
    return 0;
}

MED_IMG_END_NAMESPACE