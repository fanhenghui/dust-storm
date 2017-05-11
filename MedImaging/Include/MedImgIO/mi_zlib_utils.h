#ifndef MED_IMAGING_ZLIB_UTILS_H_
#define MED_IMAGING_ZLIB_UTILS_H_

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class IO_Export ZLibUtils
{
public:
    static IOStatus compress(const std::string& src_path , const std::string& dst_path);
    static IOStatus decompress(const std::string& src_path , const std::string& dst_path);

    static IOStatus decompress(const std::string& src_path , char* dst_buffer , unsigned int out_size);
protected:
private:
};

MED_IMAGING_END_NAMESPACE


#endif