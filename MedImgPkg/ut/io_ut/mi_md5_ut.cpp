#include <iostream>
#include "io/mi_md5.h"
#include "io/mi_io_logger.h"

using namespace medical_imaging;

int md5_ut(int argc, char* argv[]) {
    const int case_count = 2;
    const std::string cases[] = {
        "xiaoming_19900101_m",
        "cengjingfu_19570101_f"
    };

    const std::string res[] = {
        "9055ac20fcd10af8981a72c61726e20e",
        "b5c2577184b8f9c7dd8b488fdf9487ea"
    };
    
    for (int i = 0; i < case_count; ++i) {
        char md5[32] = { 0 };
        MD5::digest(cases[i], md5);
        bool pass = true;
        for (int j = 0; j < 32; ++j) {
            if (res[i][j] != md5[j]) {
                pass = false;
                break;
            }
        }
        if (pass) {
            MI_IO_LOG(MI_INFO) << "case " << i << " data: " << cases[i] << " md5: " << res[i] << "  PASS.";;
        } else {
            MI_IO_LOG(MI_ERROR) << "case " << i << " data: " << cases[i] << " md5: " << res[i] << "  FAILED";
        }
    }
    return 0;
}