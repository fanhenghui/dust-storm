#ifndef MEDIMGUTIL_MI_FILE_UTIL_H
#define MEDIMGUTIL_MI_FILE_UTIL_H

#include "MedImgCommon/mi_common_stdafx.h"
#include <string>
#include <vector>

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export FileUtil {
public:
    static void get_all_file_recursion(
        const std::string& root , const std::vector<std::string>& postfix ,
        std::vector<std::string>& files);

    static void write_raw(const std::string& path , void* buffer , unsigned int length);

protected:
private:
};

MED_IMAGING_END_NAMESPACE


#endif