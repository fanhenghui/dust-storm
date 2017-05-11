#include "MedImgIO/mi_zlib_utils.h"

using namespace medical_imaging;

void IOUT_ZlibUtils()
{
    int do_compressed = 3;
    if (0 == do_compressed)
    {
        std::string src_file("D:/temp/LKDS-00012.raw");
        std::string dst_file("D:/temp/LKDS-00012_utils.zraw");

        IOStatus status = ZLibUtils::compress(src_file , dst_file);
        if (status == IO_SUCCESS)
        {
            std::cout << "compress done.\n";
        }
        else
        {
            std::cout << "compress failed! \n";
        }
    }
    else if(1 == do_compressed)
    {
        std::string src_file("D:/temp/LKDS-00012.zraw");
        std::string dst_file("D:/temp/LKDS-00012_utils.raw");

        IOStatus status = ZLibUtils::decompress(src_file , dst_file);
        if (status == IO_SUCCESS)
        {
            std::cout << "decompress done.\n";
        }
        else
        {
            std::cout << "decompress failed! \n";
        }
    }
    else
    {
        std::string src_file("D:/temp/LKDS-00012.zraw");
        const unsigned int size = 512*512*264*2;
        char* raw_data = new char[size];

        IOStatus status = ZLibUtils::decompress(src_file , raw_data , size);
        if (status == IO_SUCCESS)
        {
            std::cout << "decompress done.\n";

            std::string dst_file("D:/temp/LKDS-00012_utils_buffer.raw");
            std::ofstream out(dst_file.c_str() , std::ios::out | std::ios::binary);
            if (out.is_open())
            {
                out.write(raw_data , size);
                out.close();
                delete [] raw_data;
            }
        }
        else
        {
            std::cout << "decompress failed! \n";
        }
    }
    
}