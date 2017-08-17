#ifndef JPEG_PARSET_H_
#define JPEG_PARSET_H_

#include "MedImgIO/mi_io_stdafx.h"
#include <string>

MED_IMAGING_BEGIN_NAMESPACE

class JpegParser
{
public:
    static IO_Export void write_to_jpeg(std::string& file, char* img_buf , int width , int height);

protected:
private:
};

MED_IMAGING_END_NAMESPACE
#endif
