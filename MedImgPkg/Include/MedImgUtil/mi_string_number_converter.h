#ifndef MEDUTIL_MI_STRING_NUMBER_CONVERTER_H
#define MEDUTIL_MI_STRING_NUMBER_CONVERTER_H

#include "MedImgUtil/mi_util_export.h"

#include <string>
#include <sstream>
#include <iomanip>

MED_IMG_BEGIN_NAMESPACE


template<class T>
class StrNumConverter {
public:
    T to_num(std::string s) {
        std::stringstream ss(s);
        T v;
        ss >> v;
        return v;
    }

    std::string to_string_decimal(T i, int precision) {
        std::stringstream ss;
        ss << std::setprecision(precision) <<  std::fixed << i;
        return ss.str();
    }

    std::string to_string(T i) {
        std::stringstream ss;
        ss << i;
        return ss.str();
    }
};


MED_IMG_END_NAMESPACE
#endif