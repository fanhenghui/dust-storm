#ifndef MED_IMG_FILE_UTIL_H_
#define MED_IMG_FILE_UTIL_H_

#include "MedImgUtil/mi_util_export.h"
#include <string>
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class Util_Export FileUtil
{
 public:
    static void get_all_file_recursion(
        const std::string& root ,const std::vector<std::string>& postfix , std::vector<std::string>& files);

 protected:
 private:
};

MED_IMG_END_NAMESPACE


#endif