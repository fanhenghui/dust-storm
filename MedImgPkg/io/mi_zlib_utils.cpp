#include "mi_zlib_utils.h"

#ifdef WIN32
#include "zlib/zconf.h"
#include "zlib/zlib.h"
#else
#include "zconf.h"
#include "zlib.h"
#include <string.h>
#endif

MED_IMG_BEGIN_NAMESPACE

#define CHUNK 16384

// Code from : http://www.zlib.net/zlib_how.html
IOStatus ZLibUtils::compress(const std::string& src_path,
                             const std::string& dst_path) {
    std::ifstream src_file(src_path.c_str(), std::ios::in | std::ios::binary);
    std::ofstream dst_file(dst_path.c_str(), std::ios::out | std::ios::binary);

    if (!src_file.is_open() || !dst_file.is_open()) {
        src_file.close();
        dst_file.close();
        return IO_FILE_OPEN_FAILED;
    }

    int ret, flush;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, Z_DEFAULT_COMPRESSION);

    if (ret != Z_OK) {
        return IO_UNSUPPORTED_YET;
    }

    /* compress until end of file */
    do {
        src_file.read((char*)in, CHUNK);

        strm.avail_in = static_cast<unsigned int>(src_file.gcount());
        strm.next_in = in;

        flush = src_file.eof() ? Z_FINISH : Z_NO_FLUSH;

        /* run deflate() on input until output buffer not full, finish
           compression if all of source has been read in */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = deflate(&strm, flush); /* no bad return value */
            have = CHUNK - strm.avail_out;

            dst_file.write((char*)out, have);
        } while (strm.avail_out == 0);

        /* done when last data in file processed */
    } while (flush != Z_FINISH);

    /* clean up and return */
    deflateEnd(&strm);

    return IO_SUCCESS;
}

IOStatus ZLibUtils::decompress(const std::string& src_path,
                               const std::string& dst_path) {
    std::ifstream src_file(src_path.c_str(), std::ios::in | std::ios::binary);
    std::ofstream dst_file(dst_path.c_str(), std::ios::out | std::ios::binary);

    if (!src_file.is_open() || !dst_file.is_open()) {
        src_file.close();
        dst_file.close();
        return IO_FILE_OPEN_FAILED;
    }

    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);

    if (ret != Z_OK) {
        return IO_UNSUPPORTED_YET;
    }

    /* decompress until deflate stream ends or end of file */
    // unsigned int current_size = 0;
    do {
        src_file.read((char*)in, CHUNK);
        strm.avail_in = static_cast<unsigned int>(src_file.gcount());

        // if (strm.avail_in != 16384)
        //{
        //    std::cout << strm.avail_in << std::endl;
        //}
        if (strm.avail_in == 0) {
            break;
        }

        strm.next_in = in;

        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);

            if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR ||
                    ret == Z_MEM_ERROR) { //��ѹ������ĩβ����Z_OK
                src_file.close();
                dst_file.close();
                inflateEnd(&strm);
                return IO_DATA_DAMAGE;
            }

            have = CHUNK - strm.avail_out;
            // current_size += have;
            dst_file.write((char*)out, have);
        } while (strm.avail_out == 0);

    } while (ret != Z_STREAM_END);

    /* clean up and return */
    inflateEnd(&strm);

    // std::cout << current_size << "\n";

    return IO_SUCCESS;
}

IOStatus ZLibUtils::decompress(const std::string& src_path, char* dst_buffer,
                               unsigned int out_size) {
    std::ifstream src_file(src_path.c_str(), std::ios::in | std::ios::binary);

    if (!src_file.is_open()) {
        src_file.close();
        return IO_FILE_OPEN_FAILED;
    }

    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);

    if (ret != Z_OK) {
        return IO_UNSUPPORTED_YET;
    }

    /* decompress until deflate stream ends or end of file */
    unsigned int current_size = 0;

    do {
        src_file.read((char*)in, CHUNK);
        strm.avail_in = static_cast<unsigned int>(src_file.gcount());

        if (strm.avail_in == 0) {
            break;
        }

        strm.next_in = in;

        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);

            if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR ||
                    ret == Z_MEM_ERROR) { //��ѹ������ĩβ����Z_OK
                src_file.close();
                inflateEnd(&strm);
                return IO_DATA_DAMAGE;
            }

            have = CHUNK - strm.avail_out;

            current_size += have;

            if (current_size > out_size) {
                src_file.close();
                inflateEnd(&strm);
                return IO_DATA_DAMAGE;
            }

            memcpy(dst_buffer + (current_size - have), out, have);
        } while (strm.avail_out == 0);
    } while (ret != Z_STREAM_END);

    /* clean up and return */
    inflateEnd(&strm);

    if (current_size != out_size) {
        return IO_DATA_DAMAGE;
    }

    return IO_SUCCESS;
}

MED_IMG_END_NAMESPACE