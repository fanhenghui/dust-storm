#include "mi_time_util.h"
#include <exception>
#include <time.h>
#include <sstream>

MED_IMG_BEGIN_NAMESPACE

static const int DAY_LIMIT[12] = {31,28,31,30,31,30,31,31,30,31,30,31};

inline bool check_num(char num) {
    return (num >= '0' && num <= '9');
}

inline bool check_leap_year(int y) {
    return (((y%4)==0)&&(y%100)!=0)||(y%400==0);
}

int TimeUtil::check_yyyymmdd(const std::string& date) {
    if (date.size() != 8) {
        return -1;
    }
    for (int i=0; i<8; ++i) {
        if (!check_num(date[i])) {
            return -1;
        }
    }
    const int month = atoi(date.substr(4,2).c_str());
    if (month < 1 || month > 12) {
        return -1;
    }
    const int day = atoi(date.substr(6,2).c_str());
    if (month == 2) {
        const int year = atoi(date.substr(0,4).c_str());
        const int day_limit = check_leap_year(year) ? 29:28;
        if (day < 1 || day > day_limit) {
            return -1;
        }
    } else {
        if (day < 1 || day > DAY_LIMIT[month-1]) {
            return -1;
        }
    }

    return 0;
}

int TimeUtil::check_yyyymmdd_range(const std::string& date_range) {
    if (date_range.size() != 17) {
        return -1;
    }
    const std::string date0 = date_range.substr(0,8);
    const std::string date1 = date_range.substr(9,8);
    if (date_range[8] != '-') {
        return -1;
    }
    if (TimeUtil::check_yyyymmdd(date0) && TimeUtil::check_yyyymmdd(date1)) {
        return 0;
    } else {
        return -1;
    }
}

int TimeUtil::check_hhmmss(const std::string& time) {
    if (time.size() != 6) {
        return -1;
    }
    for (int i=0; i<6; ++i) {
        if (!check_num(time[i])) {
            return -1;
        }
    }
    const int h = atoi(time.substr(0,2).c_str());
    if (h < 0 || h > 23) {
        return -1;
    }
    const int m = atoi(time.substr(2,2).c_str());
    if (m < 0 || m > 59) {
        return -1;
    }
    const int s = atoi(time.substr(4,2).c_str());
    if (s < 0 || s > 59) {
        return -1;
    }

    return 0;
}

int TimeUtil::check_hhmmssfrac(const std::string& time) {
    if (time.size() < 7) {
        return -1;
    }
    if (time[6] != '.') {
        return -1;
    }
    return TimeUtil::check_hhmmss(time.substr(0,6));
}

int TimeUtil::remove_time_frac(std::string& time) {
    if (time.size() < 7) {
        return -1;
    }
    time = time.substr(0,6);
    return 0;
}

int TimeUtil::check_hhmmss_range(const std::string& time_range) {
    if (time_range.size() != 13) {
        return -1;
    }
    const std::string time0 = time_range.substr(0,6);
    const std::string time1 = time_range.substr(7,6);
    if (time_range[6] != '-') {
        return -1;
    }
    if (TimeUtil::check_hhmmss(time0) && TimeUtil::check_hhmmss(time1)) {
        return 0;
    } else {
        return -1;
    }
}

int TimeUtil::check_yyyymmddhhmmss(const std::string& datetime) {
    if (datetime.size() != 14) {
        return -1;
    }
    const std::string& date = datetime.substr(0,8);
    if (-1 == check_yyyymmdd(date)) {
        return -1;
    }
    const std::string& time = datetime.substr(8,6);
    if (-1 == check_hhmmss(time)) {
        return -1;
    }

    return 0;
}

std::string TimeUtil::current_date(int time_zone) {
    time_zone = time_zone < 0 ? 0 : time_zone; 
    time_zone = time_zone > 23 ? 23 : time_zone;
    time_t rt = time(0) + time_zone*3600;
    tm* lt = gmtime(&rt);

    std::stringstream ss;
    ss << lt->tm_year + 1900;
    if (lt->tm_mon < 9) {
        ss << 0;
    }
    ss << lt->tm_mon + 1;
    if (lt->tm_mday < 10) {
        ss << 0;
    }
    ss << lt->tm_mday;
    return ss.str();
}

std::string TimeUtil::current_time(int time_zone) {
    time_zone = time_zone < 0 ? 0 : time_zone; 
    time_zone = time_zone > 23 ? 23 : time_zone;
    time_t rt = time(0) + time_zone*3600;
    tm* lt = gmtime(&rt);

    std::stringstream ss;
    if (lt->tm_hour < 10) {
        ss << 0;
    }
    ss << lt->tm_hour;
    if (lt->tm_min < 10) {
        ss << 0;
    }
    ss << lt->tm_min;
    if (lt->tm_sec < 10) {
        ss << 0;
    }
    ss << lt->tm_sec;
    return ss.str();
}

std::string TimeUtil::current_date_time(int time_zone) {
    time_zone = time_zone < 0 ? 0 : time_zone; 
    time_zone = time_zone > 23 ? 23 : time_zone;
    time_t rt = time(0) + time_zone*3600;
    tm* lt = gmtime(&rt);
    
    std::stringstream ss;
    ss << lt->tm_year + 1900;
    if (lt->tm_mon < 9) {
        ss << 0;
    }
    ss << lt->tm_mon + 1;
    if (lt->tm_mday < 10) {
        ss << 0;
    }
    ss << lt->tm_mday;

    if (lt->tm_hour < 10) {
        ss << 0;
    }
    ss << lt->tm_hour;
    if (lt->tm_min < 10) {
        ss << 0;
    }
    ss << lt->tm_min;
    if (lt->tm_sec < 10) {
        ss << 0;
    }
    ss << lt->tm_sec;
    return ss.str();
}

MED_IMG_END_NAMESPACE