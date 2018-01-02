#ifndef MEDIMG_IO_MI_MD5_H
#define MEDIMG_IO_MI_MD5_H

#include <string>
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export MD5
{
public:
    static int digest(const std::string& data, unsigned char(&md5_value)[16]);
    static int digest(const std::string& data, char(&hex_str)[32]);
};

MED_IMG_END_NAMESPACE


#endif
