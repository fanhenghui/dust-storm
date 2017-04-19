#ifndef MED_IMAGING_STRING_NUMBER_CONVERTER_H_
#define MED_IMAGING_STRING_NUMBER_CONVERTER_H_

#include "MedImgCommon/mi_common_stdafx.h"

#include <string>
#include <sstream>
#include <iomanip>

MED_IMAGING_BEGIN_NAMESPACE

template<class T>
class StrNumConverter
{
public:
    T ToNumber(std::string s)
    {
        std::stringstream ss(s);
        T v;
        ss >> v;
        return v;
    }

    std::string ToStringDecimal( T i, int iPrecision)
    {
        std::stringstream ss;
        ss << std::setprecision(iPrecision) <<  std::fixed << i;
        return ss.str();
    }

    std::string ToString(T i)
    {
        std::stringstream ss;
        ss << i;
        return ss.str();
    }
};


MED_IMAGING_END_NAMESPACE
#endif