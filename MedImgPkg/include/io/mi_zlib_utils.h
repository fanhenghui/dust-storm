#ifndef MED_IMG_ZLIB_UTILS_H_
#define MED_IMG_ZLIB_UTILS_H_

#include "io/mi_io_export.h"
#include "io/mi_io_define.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export ZLibUtils
{
public:
    static IOStatus compress(const std::string& src_path , const std::string& dst_path);
    static IOStatus decompress(const std::string& src_path , const std::string& dst_path);

    static IOStatus decompress(const std::string& src_path , char* dst_buffer , unsigned int out_size);
protected:
private:
};

MED_IMG_END_NAMESPACE


#endif