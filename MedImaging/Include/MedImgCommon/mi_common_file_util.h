#ifndef MEDIMGUTIL_MI_FILE_UTIL_H
#define MEDIMGUTIL_MI_FILE_UTIL_H

#include "MedImgCommon/mi_common_stdafx.h"
#include <string>
#include <vector>
#include <map>
#include <set>

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export FileUtil {
public:
    static void get_all_file_recursion(
        const std::string& root , const std::set<std::string>& postfix ,
        std::vector<std::string>& files);

    static void get_all_file_recursion(
        const std::string& root , const std::set<std::string>& postfix ,
        std::map<std::string , std::vector<std::string>>& files);

    static int write_raw(const std::string& path , void* buffer , unsigned int length);

protected:
private:
};

MED_IMAGING_END_NAMESPACE


#endif