#ifndef MED_IMG_UTIL_MI_TIME_UTIL_H
#define MED_IMG_UTIL_MI_TIME_UTIL_H

#include "util/mi_util_export.h"
#include <string>

MED_IMG_BEGIN_NAMESPACE

class Util_Export TimeUtil {
public:
    static int check_yyyymmdd(const std::string& date);
    static int check_yyyymmdd_range(const std::string& date_range);
    static int check_hhmmss(const std::string& time);
    static int check_hhmmss_range(const std::string& time_range);
    static int check_yyyymmddhhmmss(const std::string& datetime);
};

MED_IMG_END_NAMESPACE

#endif