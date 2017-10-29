#include "io/mi_targa_parser.h"
#include "io/mi_jpeg_parser.h"
#include "util/mi_file_util.h"
#include <string>
#include <iostream>
#include <sstream>

using namespace medical_imaging;

int targa_ut(int argc, char* argv[]) {
    if (argc != 2) {
        return -1;
    }

    std::string file(argv[1]);

    int width,height,channel;
    unsigned char* img_buf = nullptr;
    if (IO_SUCCESS != TargaParser::load(file, width, height, channel, img_buf)) {
        return -1;
    }

    if (channel == 4){ 
        std::cout << "channel is 4.\n";
        unsigned char* img_buf2 = img_buf;
        img_buf = new unsigned char[width*height*3];
        for (int i = 0; i< width*height; ++i) {
            img_buf[i*3] = img_buf2[i*4];
            img_buf[i*3+1] = img_buf2[i*4+1];
            img_buf[i*3+2] = img_buf2[i*4+2];
        }
        delete [] img_buf2;
    } else {
        std::stringstream ss;
        ss << "/home/wangrui22/data/navi_" << width << "_" << height << "_" << channel << ".raw";
        FileUtil::write_raw(ss.str(), img_buf, width*height*channel);
    }

    std::string file2 = "/home/wangrui22/data/targa.jpeg";
    JpegParser::write_to_jpeg(file2, (char*)img_buf, width, height);

    delete [] img_buf;
    return 0;

}