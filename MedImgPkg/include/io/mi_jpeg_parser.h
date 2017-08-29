#ifndef MEDIMGIO_JPEG_PARSET_H
#define MEDIMGIO_JPEG_PARSET_H

#include "io/mi_io_export.h"
#include <string>

MED_IMG_BEGIN_NAMESPACE

class JpegParser {
public:
    static IO_Export void write_to_jpeg(std::string& file, char* img_buf , int width , int height);

protected:
private:
};

MED_IMG_END_NAMESPACE
#endif
