#include "mi_targa_parser.h"
#include <fstream>
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

IOStatus TargaParser::load(const std::string& file, int& width, int& height, int& channel, unsigned char*& img_buf) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        in.close();
        MI_IO_LOG(MI_ERROR) << "open targa file " << file << " failed.";
        return IO_FILE_OPEN_FAILED;
    }

    TGAHEADER tga_header;
    unsigned int img_size = 0;
    short depth = 0;
    width = 0;
    height = 0;
    channel = 3;

    // attempt to open the file
    unsigned char head[18];
    in.read((char*)(&head), 18);
    tga_header.identsize = head[0];
    tga_header.colorMapType = head[1];
    tga_header.imageType = head[2];
    tga_header.colorMapStart = ((unsigned short)(head[4])) << 8 | (head[3]);
    tga_header.colorMapLength = ((unsigned short)(head[6])) << 8 | (head[5]);
    tga_header.colorMapBits = head[7];
    tga_header.xstart = ((unsigned short)(head[9]) << 8 | (head[8]));
    tga_header.ystart = ((unsigned short)(head[11]) << 8 | (head[10]));
    tga_header.width = ((unsigned short)(head[13]) << 8 | (head[12]));
    tga_header.height = ((unsigned short)(head[15]) << 8 | (head[14]));
    tga_header.bits = head[16];
    tga_header.descriptor = head[17];

    width = tga_header.width;
    height = tga_header.height;
    channel = tga_header.bits / 8;

    // only support 8, 24, or 32 bit targa's.
    if(tga_header.bits != 8 && tga_header.bits != 24 && tga_header.bits != 32) {
        MI_IO_LOG(MI_ERROR) << "just support bits 8 24 32 targa image.";
        return IO_UNSUPPORTED_YET;
    }

    // Calculate size of image buffer
    img_size = tga_header.width * tga_header.height * channel;

    // Allocate memory and check for success
    if (img_buf != nullptr) {
        MI_IO_LOG(MI_WARNING) << "input image buffer is not null. Maybe cause mem leak.";
    }
    img_buf = new unsigned char[img_size];
    if (img_buf == nullptr) {
        MI_IO_LOG(MI_FATAL) << "not enough memory to allocate for targa.";
        return IO_NOT_ENOUGH_MEMORY;
    }

    in.read((char*)img_buf, img_size);
    in.close();
    MI_IO_LOG(MI_TRACE) << "load targa file " << file << " success.";
    return IO_SUCCESS;
}

MED_IMG_END_NAMESPACE