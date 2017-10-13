#include <iostream>
#include <fstream>
#include <string>
#include "util/mi_file_util.h"
#include "arithmetic/mi_run_length_operator.h"

using namespace medical_imaging;

int main(int argc , char* argv[]) {
    if (argc < 2) {
        std::cout << "mask to be compressed path is null.\n";
        return -1;
    }

    char* mask = nullptr;
    unsigned int length = 0;
    if (0 != FileUtil::read_raw_ext(argv[1], mask, length) ) {
        return -1;
    }
    std::vector<unsigned int> res = RunLengthOperator::encode((unsigned char*)mask, length);

    FileUtil::write_raw(std::string(argv[1]) + ".rle", res.data(), res.size()*sizeof(unsigned int));

    std::cout << "done\n";
    return 0;

}