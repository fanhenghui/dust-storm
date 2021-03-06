#ifndef MEDIMGUTIL_MI_FILE_UTIL_H
#define MEDIMGUTIL_MI_FILE_UTIL_H

#include "util/mi_util_export.h"
#include <string>
#include <vector>
#include <set>
#include <map>

MED_IMG_BEGIN_NAMESPACE

class Util_Export FileUtil {
public:
    static int get_all_file_recursion(
        const std::string& root, const std::set<std::string>& postfix,
        std::vector<std::string>& files);
    
    static int get_all_file_recursion(
        const std::string& root, const std::set<std::string>& postfix,
        std::map<std::string, std::vector<std::string>>& files);

    static float get_size_mb(std::vector<std::string>& files);

    static int check_direction(const std::string& path);
    static int make_direction(const std::string& path);

    static int write_raw(const std::string& path, void* buffer, unsigned int length);
    static int read_raw(const std::string& path, void* buffer, unsigned int length);
    static int read_raw_ext(const std::string& path, char*& buffer, unsigned int& length);

    static int get_file_size(const std::string& path, int64_t& fsize);

    static int remove_file(const std::string& path);
    static int copy_file(const std::string& src, const std::string& dst);

protected:
private:
};

MED_IMG_END_NAMESPACE


#endif