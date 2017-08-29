#ifndef MEDIMGIO_ZLIB_UTILS_H
#define MEDIMGIO_ZLIB_UTILS_H

#include "io/mi_io_define.h"
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export ZLibUtils {
public:
    static IOStatus compress(const std::string& src_path,
                             const std::string& dst_path);
    static IOStatus decompress(const std::string& src_path,
                               const std::string& dst_path);

    static IOStatus decompress(const std::string& src_path, char* dst_buffer,
                               unsigned int out_size);

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif