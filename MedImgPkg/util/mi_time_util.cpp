#include "mi_time_util.h"
#include <exception>
#include "mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

static const int DAY_LIMIT[12] = {31,28,31,30,31,30,31,31,30,31,30,31};

inline bool check_num(char num) {
    return (num >= '0' && num <= '9');
}

inline bool check_leap_year(int y) {
    return (((y%4)==0)&&(y%100)!=0)||(y%400==0);
}

int TimeUtil::check_yyyymmdd(const std::string& date) {
    try {
        if (date.size() != 8) {
            throw std::exception(std::logic_error("invalid length."));
        }

        for (int i=0; i<8; ++i) {
            if (!check_num(date[i])) {
                throw std::exception(std::logic_error("not number."));
            }
        }

        const int month = atoi(date.substr(4,2).c_str());
        if (month < 1 || month > 12) {
            throw std::exception(std::logic_error("invalid month."));  
        }

        const int day = atoi(date.substr(6,2).c_str());
        if (month == 2) {
            const int year = atoi(date.substr(0,4).c_str());
            const int day_limit = check_leap_year(year) ? 29:28;
            if (day < 1 || day > day_limit) {
                throw std::exception(std::logic_error("invalid day."));
            }
        } else {
            if (day < 1 || day > DAY_LIMIT[month-1]) {
                throw std::exception(std::logic_error("invalid day."));
            }
        }

    } catch (const std::exception& e) {
        MI_UTIL_LOG(MI_ERROR) << "invalid date: " << date << ", " << e.what();
        return -1;
    }

    return 0;
}

int TimeUtil::check_hhmmss(const std::string& time) {
    try {
        if (time.size() != 6) {
            throw std::exception(std::logic_error("invalid length."));
        }
        for (int i=0; i<8; ++i) {
            if (!check_num(time[i])) {
                throw std::exception(std::logic_error("not number."));
            }
        }

        const int h = atoi(time.substr(0,2).c_str());
        if (h < 0 || h > 23) {
            throw std::exception(std::logic_error("invalid hour."));
        }

        const int m = atoi(time.substr(2,2).c_str());
        if (m < 0 || m > 59) {
            throw std::exception(std::logic_error("invalid minute."));
        }

        const int s = atoi(time.substr(4,2).c_str());
        if (s < 0 || s > 59) {
            throw std::exception(std::logic_error("invalid second."));
        }


    } catch (const std::exception& e) {
        MI_UTIL_LOG(MI_ERROR) << "invalid time: " << time << ", " << e.what();
        return -1;
    }
}

int TimeUtil::check_yyyymmddhhmmss(const std::string& datetime) {
    try {
        if (datetime.size() != 14) {
            throw std::exception(std::logic_error("invalid size."));
        }
        const std::string& date = datetime.substr(0,8);
        if (-1 == check_yyyymmdd(date)) {
            throw std::exception(std::logic_error("invalid date."));
        }
        const std::string& time = datetime.substr(8,6);
        if (-1 == check_hhmmss(time)) {
            throw std::exception(std::logic_error("invalid time."));
        }
    } catch (const std::exception& e) {
        MI_UTIL_LOG(MI_ERROR) << "invalid datetime: " << datetime << ", " << e.what();
        return -1;
    } 
}

// int64_t TimeUtil::date_to_timestamp(const std::string& date) {

// }

// int64_t TimeUtil::datetime_to_timestamp(const std::string& datetime) {

// }

MED_IMG_END_NAMESPACE